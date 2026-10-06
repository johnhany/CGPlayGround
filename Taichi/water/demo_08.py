"""Splash: stones and clicks burst into spray, foam, and ring ripples.

The dive scene gains fixed-step ballistic drops, a parameterized water crown,
volume-balanced ripple impulses, and atomic landing foam. Transparent water
and solid occlusion are handled separately. Droplets use refraction-aware
projection and weighted HDR coverage before shared bloom and tonemapping.

Run: uv run python water/demo_08.py
Export: uv run python water/demo_08.py --headless --output output/splash.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .demo_06 import (SEA_DX, SEA_HALF, SEA_N, SIM_DT, sea_parameters,
                          shore_terrain, simulation_budget, water_pick)
    from .demo_07 import (DIVE_PRESETS, DiveCamera, DiveRenderer,
                          bed_height_py, IOR)
else:
    from demo_06 import (SEA_DX, SEA_HALF, SEA_N, SIM_DT, sea_parameters,
                         shore_terrain, simulation_budget, water_pick)
    from demo_07 import (DIVE_PRESETS, DiveCamera, DiveRenderer,
                         bed_height_py, IOR)

__all__ = ["SPLASH_PRESETS", "SplashRenderer", "advect_drop_cpu", "apply_splash_preset",
           "main", "parse_args", "sea_parameters", "simulation_budget", "water_pick",
           "DROP_CAP", "CROWN_CAP", "DRAG", "GRAVITY"]

DROP_CAP = 2048
CROWN_BANDS = 8
CROWN_SEGMENTS = 48
CROWN_CAP = CROWN_BANDS * CROWN_SEGMENTS
GRAVITY = 9.81
# Quadratic drag: dv/dt = -DRAG * |v| * v, integrated implicitly per step.
DRAG = 0.08

SPLASH_PRESETS = dict(DIVE_PRESETS)
SPLASH_PRESETS["impact"] = ((-22.0, 0.45, 0.0), (0.95, 0.24, 7.5), 52.0)
SPLASH_PRESETS["top"] = ((-22.0, 0.0, 0.0), (0.95, 1.42, 12.0), 55.0)
SPLASH_PRESETS["overview"] = ((-22.0, 0.0, 0.0), (0.95, 0.55, 20.0), 48.0)


def apply_splash_preset(camera, name):
    target, (yaw, pitch, distance), fov = SPLASH_PRESETS[name]
    camera.target = np.array(target)
    camera.yaw, camera.pitch, camera.distance = yaw, pitch, distance
    camera.fov = fov
    camera.lift_above_bed()


def advect_drop_cpu(position, velocity, dt):
    """Python mirror of the droplet integrator for tests and tools."""
    position = np.asarray(position, dtype=float).copy()
    velocity = np.asarray(velocity, dtype=float).copy()
    velocity[1] -= GRAVITY * dt
    speed = float(np.linalg.norm(velocity))
    velocity /= 1.0 + DRAG * speed * dt
    position += velocity * dt
    return position, velocity


@ti.data_oriented
class SplashRenderer(DiveRenderer):
    """Dive renderer plus a ballistic splash system on the shallow-water grid."""

    def __init__(self, width, height, samples=4):
        super().__init__(width, height, samples)
        # x: splat brightness at composite, y: foam crown strength,
        # z: trail length, w: spare.
        self.splash_controls = ti.Vector.field(4, ti.f32, shape=())
        self.splash_controls[None] = [1.0, 1.0, 0.6, 0.0]
        self.drop_pos = ti.Vector.field(3, ti.f32, shape=DROP_CAP)
        self.drop_prev = ti.Vector.field(3, ti.f32, shape=DROP_CAP)
        self.drop_vel = ti.Vector.field(3, ti.f32, shape=DROP_CAP)
        self.drop_normal = ti.Vector.field(3, ti.f32, shape=DROP_CAP)
        # age, life, size (m), seed
        self.drop_aux = ti.Vector.field(4, ti.f32, shape=DROP_CAP)
        self.drop_alive = ti.field(ti.i32, shape=DROP_CAP)
        self.drop_land_flag = ti.field(ti.i32, shape=DROP_CAP)
        # Landing x/z and ripple strength, valid while the flag is set.
        self.drop_land = ti.Vector.field(3, ti.f32, shape=DROP_CAP)
        self.drop_cursor = ti.field(ti.i32, shape=())
        self.splash_accum = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.splash_weight = ti.field(ti.f32, shape=(width, height))
        self.foam_delta = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.pulse_norm = ti.Vector.field(2, ti.f32, shape=())
        self.drop_kind = ti.field(ti.i32, shape=DROP_CAP)
        self.drop_origin = ti.Vector.field(4, ti.f32, shape=DROP_CAP)
        self.drop_clearance = ti.field(ti.f32, shape=DROP_CAP)
        self.drop_path = ti.Vector.field(4, ti.f32, shape=DROP_CAP)
        self.drop_visibility = ti.field(ti.f32, shape=DROP_CAP)
        self.particle_samples = ti.field(ti.i32, shape=())
        self.particle_samples[None] = samples
        self.rng = np.random.default_rng(808)
        self.stone_sinking = False
        self.stone_sink_age = 0.0
        # The falling stone is an animated rounded box from the base scene.
        self.stone_index = self.boxes
        self.box_center[self.stone_index] = [0.0, -60.0, 0.0]
        self.box_extent[self.stone_index] = [0.001, 0.001, 0.001]
        self.box_bevel[self.stone_index] = 0.0008
        self.box_color[self.stone_index] = [0.16, 0.17, 0.18]
        self.box_material[self.stone_index] = 4
        self.boxes += 1
        self.stone_pos = None
        self.stone_vy = 0.0
        self.stone_wait = 1.2
        self.reset_drops()

    def reset_drops(self):
        self.drop_alive.fill(0)
        self.drop_land_flag.fill(0)
        self.splash_accum.fill(0)
        self.splash_weight.fill(0)
        self.drop_kind.fill(0)
        self.stone_sinking = False
        self.stone_sink_age = 0.0
        self.drop_cursor[None] = 0
        self.stone_pos = None
        self.stone_vy = 0.0
        self.stone_wait = 1.2
        self.box_center[self.stone_index] = [0.0, -60.0, 0.0]
        self.box_extent[self.stone_index] = [0.001, 0.001, 0.001]
        self.box_bevel[self.stone_index] = 0.0008

    @ti.kernel
    def surface_probe(self, x: ti.f32, z: ti.f32, clock: ti.f32) -> ti.f32:
        return self.sea.sample(x, z, clock).x

    @ti.kernel
    def _burst_drops(self, cx: ti.f32, cz: ti.f32, clock: ti.f32,
                     count: ti.i32, energy: ti.f32):
        cursor = self.drop_cursor[None]
        for k in range(count):
            base = ti.Vector([cx * .73 + .11, cz * 1.19 + .37]) + .6180339*k
            a1, a2 = self.hash2(base), self.hash2(base+ti.Vector([13.7,5.1]))
            a3, a4 = self.hash2(base+ti.Vector([27.3,11.7])), self.hash2(base+ti.Vector([41.9,23.3]))
            angle = a1*2*math.pi
            radial, vy, vr, size, kind = .22+.48*a2, 2.5+2*a3, 1.+1.6*a4, .025+.030*a4, 0
            if k%10 < 2:
                radial,vy,vr,size,kind = .5+.35*a2,1.8+1.4*a3,2.+1.5*a4,.007+.008*a4,1
            elif k%10 == 2:
                radial,vy,vr,size,kind = .04+.12*a2,4.5+1.4*a3,.25+.45*a4,.035+.012*a4,3
            lift = ti.pow(ti.min(1.6,energy),.55)
            direction = ti.Vector([ti.cos(angle),0.,ti.sin(angle)])
            idx = (cursor+k)%DROP_CAP
            p = ti.Vector([cx,0.,cz])+direction*radial
            p.y = self.wave_height(p.x,p.z,clock)+.055
            self.drop_pos[idx],self.drop_prev[idx] = p,p
            self.drop_vel[idx] = direction*(vr*lift)+ti.Vector([0.,vy*lift,0.])
            self.drop_normal[idx] = ti.Vector([.2*direction.x,1.,.2*direction.z]).normalized()
            self.drop_aux[idx] = ti.Vector([-.08*a2,2.0+2*vy*lift/GRAVITY,size,a4])
            self.drop_alive[idx],self.drop_land_flag[idx],self.drop_kind[idx] = 1,0,kind
            self.drop_clearance[idx] = .055
        self.drop_cursor[None] = (cursor+count)%DROP_CAP

    @ti.func
    def crown_position(self,i,clock):
        cx,cz,energy,band = self.drop_origin[i]
        theta = self.drop_aux[i].w
        age = ti.max(0.,self.drop_aux[i].x)
        phase = ti.min(1.,age/self.drop_aux[i].y)
        height = (1.05+.35*energy)*ti.sin(math.pi*phase)
        rim = .52+1.1*age
        radius = .28+(rim-.28)*band
        lobes = 1+.16*ti.sin(9*theta+1.2)+.08*ti.sin(17*theta)
        x,z = cx+radius*ti.cos(theta),cz+radius*ti.sin(theta)
        y = self.wave_height(x,z,clock)+.035+height*ti.pow(band,1.3)*lobes
        return ti.Vector([x,y,z])

    @ti.kernel
    def _burst_crown(self,cx:ti.f32,cz:ti.f32,clock:ti.f32,energy:ti.f32):
        cursor = self.drop_cursor[None]
        for k in range(CROWN_CAP):
            i = (cursor+k)%DROP_CAP
            theta = (k//CROWN_BANDS)*2*math.pi/CROWN_SEGMENTS
            self.drop_origin[i] = ti.Vector([cx,cz,energy,(k%CROWN_BANDS)/(CROWN_BANDS-1)])
            self.drop_aux[i] = ti.Vector([0.,.62,.13,theta])
            self.drop_kind[i],self.drop_alive[i],self.drop_land_flag[i] = 2,1,0
            p = self.crown_position(i,clock)
            self.drop_pos[i],self.drop_prev[i] = p,p
            self.drop_vel[i] = ti.Vector([0.,0.,0.])
            self.drop_normal[i] = ti.Vector([-ti.cos(theta)*.8,.6,-ti.sin(theta)*.8])
        self.drop_cursor[None] = (cursor+CROWN_CAP)%DROP_CAP

    @ti.kernel
    def _pulse_weights(self,cx:ti.f32,cz:ti.f32,radius:ti.f32):
        self.pulse_norm[None] = ti.Vector([0.,0.])
        for i,j in self.sea.h:
            r2 = ((-SEA_HALF+i*SEA_DX-cx)**2+(-SEA_HALF+j*SEA_DX-cz)**2)/(radius*radius)
            if r2<36 and self.sea.h[i,j]>.01:
                ti.atomic_add(self.pulse_norm[None].x,ti.exp(-r2))
                ti.atomic_add(self.pulse_norm[None].y,ti.exp(-r2/4))

    @ti.kernel
    def _apply_pulse(self,cx:ti.f32,cz:ti.f32,radius:ti.f32,height:ti.f32):
        norm = self.pulse_norm[None]
        for i,j in self.sea.h:
            r2 = ((-SEA_HALF+i*SEA_DX-cx)**2+(-SEA_HALF+j*SEA_DX-cz)**2)/(radius*radius)
            if r2<36 and self.sea.h[i,j]>.01:
                self.sea.h[i,j] += height*(ti.exp(-r2)-norm.x/ti.max(1e-8,norm.y)*ti.exp(-r2/4))

    def pulse(self,x,z,height,radius=.8):
        self._pulse_weights(x,z,radius)
        self._apply_pulse(x,z,radius,height)

    @ti.kernel
    def _foam_burst(self, cx: ti.f32, cz: ti.f32, strength: ti.f32):
        gx = (cx + SEA_HALF) / SEA_DX
        gz = (cz + SEA_HALF) / SEA_DX
        reach = ti.cast(ti.ceil(2.4 / SEA_DX), ti.i32)
        ci, cj = ti.cast(ti.floor(gx), ti.i32), ti.cast(ti.floor(gz), ti.i32)
        for a, b in ti.ndrange((-reach, reach + 1), (-reach, reach + 1)):
            i, j = ci + a, cj + b
            if 0 <= i < SEA_N and 0 <= j < SEA_N:
                dx, dz = (i - gx) * SEA_DX, (j - gz) * SEA_DX
                r2 = dx * dx + dz * dz
                ring = ti.exp(-ti.pow(ti.sqrt(r2) - 0.65, 2.0) / 0.24)
                crown = ti.exp(-r2 / 0.18)
                foam = self.sea.foam[i, j] + strength * (0.85 * ring + 0.5 * crown)
                self.sea.foam[i, j] = ti.max(0.0, ti.min(1.0, foam))

    @ti.kernel
    def _advect_drops(self, dt: ti.f32, clock: ti.f32):
        for i in range(DROP_CAP):
            if self.drop_alive[i] == 1:
                p = self.drop_pos[i]
                self.drop_prev[i] = p
                age = self.drop_aux[i].x+dt
                self.drop_aux[i].x = age
                if self.drop_kind[i] == 2:
                    self.drop_pos[i] = self.crown_position(i,clock)
                    if age>=self.drop_aux[i].y:
                        self.drop_alive[i] = 0
                elif age>=0:
                    velocity = self.drop_vel[i]
                    velocity.y -= GRAVITY*dt
                    velocity /= 1+DRAG*velocity.norm()*dt
                    q = p+velocity*dt
                    surface = self.sea.sample(q.x,q.z,clock)
                    clearance = q.y-surface.x
                    previous = self.drop_clearance[i]
                    # Seeded diagnostic particles also use actual clearance.
                    if previous<=0:
                        previous = p.y-self.wave_height(p.x,p.z,clock)
                    if surface.w>1e-5 and clearance<=0 and (previous>0 or velocity.y<0):
                        fraction = ti.max(0.,ti.min(1.,previous/ti.max(1e-6,previous-clearance)))
                        hit = p+(q-p)*fraction
                        strength = ti.min(5.,ti.pow(self.drop_aux[i].z/.03,3)*ti.max(0.,-velocity.y)/6)
                        self.drop_land[i] = ti.Vector([hit.x,hit.z,strength])
                        self.drop_land_flag[i],self.drop_alive[i] = 1,0
                    elif q.y<=shore_terrain(q.x,q.z) or age>=self.drop_aux[i].y or ti.abs(q.x)>SEA_HALF or ti.abs(q.z)>SEA_HALF:
                        self.drop_alive[i] = 0
                    self.drop_pos[i],self.drop_vel[i],self.drop_clearance[i] = q,velocity,clearance

    @ti.kernel
    def _landed_ripples(self):
        for i in range(DROP_CAP):
            if self.drop_land_flag[i] == 1:
                cx,cz,strength = self.drop_land[i]
                gx,gz = (cx+SEA_HALF)/SEA_DX,(cz+SEA_HALF)/SEA_DX
                ci,cj = ti.cast(ti.floor(gx),ti.i32),ti.cast(ti.floor(gz),ti.i32)
                inner,outer = 0.,0.
                for a,b in ti.ndrange((-5,6),(-5,6)):
                    ii,jj = ci+a,cj+b
                    if 0<=ii<SEA_N and 0<=jj<SEA_N and self.sea.h[ii,jj]>.01:
                        r2 = ((ii-gx)**2+(jj-gz)**2)*SEA_DX**2/.35**2
                        inner+=ti.exp(-r2)
                        outer+=ti.exp(-r2/4)
                for a,b in ti.ndrange((-5,6),(-5,6)):
                    ii,jj = ci+a,cj+b
                    if 0<=ii<SEA_N and 0<=jj<SEA_N and self.sea.h[ii,jj]>.01:
                        r2 = ((ii-gx)**2+(jj-gz)**2)*SEA_DX**2/.35**2
                        pulse = ti.exp(-r2)-inner/ti.max(1e-8,outer)*ti.exp(-r2/4)
                        ti.atomic_add(self.sea.h[ii,jj],-.003*strength*pulse)
                        ti.atomic_add(self.foam_delta[ii,jj],.12*strength*ti.exp(-r2))
                self.drop_land_flag[i] = 0

    @ti.kernel
    def _apply_landing_foam(self):
        for i,j in self.foam_delta:
            self.sea.foam[i,j] = ti.min(1.,self.sea.foam[i,j]+self.foam_delta[i,j]*self.splash_controls[None].y)
            self.foam_delta[i,j] = 0.

    @ti.func
    def clear_segment(self,a,b):
        delta=b-a
        length=delta.norm()
        ray=delta/ti.max(1e-6,length)
        limit=ti.max(.001,length-.004)
        return ti.cast(self.occlusion_distance(a+ray*.002,ray,limit)>=limit-.0005,ti.f32)

    @ti.func
    def optical_particle(self,eye,position,clock):
        delta=position-eye
        ray=delta/ti.max(1e-6,delta.norm())
        water_length=0.
        visible=1.
        eye_wet=eye.y<self.wave_height(eye.x,eye.z,clock)
        point_wet=position.y<self.wave_height(position.x,position.z,clock)
        q=position
        if eye_wet == point_wet:
            if eye_wet:
                water_length=delta.norm()
        else:
            water,air=eye,position
            if not eye_wet:
                water,air=position,eye
            height=self.wave_height(water.x,water.z,clock)
            horizontal=ti.Vector([air.x-water.x,air.z-water.z])
            separation=horizontal.norm()
            dw,da=ti.max(.001,height-water.y),ti.max(.001,air.y-height)
            lo,hi=0.,separation
            # Exact planar Snell solution gives a stable starting point.
            for _ in range(16):
                mid=(lo+hi)*.5
                error=IOR*mid/ti.sqrt(dw*dw+mid*mid)-(separation-mid)/ti.sqrt(da*da+(separation-mid)**2)
                if error>0: hi=mid
                else: lo=mid
            uv=ti.Vector([water.x,water.z])+horizontal*(lo+hi)*.5/ti.max(1e-6,separation)
            # Minimize optical path on the resolved curved interface.
            for _ in range(10):
                surface=self.sea.sample(uv.x,uv.y,clock)
                q=ti.Vector([uv.x,surface.x,uv.y])
                aw,bw=q-water,q-air
                lw,la=ti.max(.001,aw.norm()),ti.max(.001,bw.norm())
                uw,ua=aw/lw,bw/la
                jx,jz=ti.Vector([1.,surface.y,0.]),ti.Vector([0.,surface.z,1.])
                gx,gz=IOR*uw.dot(jx)+ua.dot(jx),IOR*uw.dot(jz)+ua.dot(jz)
                hxx=IOR/lw*(jx.dot(jx)-uw.dot(jx)**2)+1/la*(jx.dot(jx)-ua.dot(jx)**2)+.01
                hzz=IOR/lw*(jz.dot(jz)-uw.dot(jz)**2)+1/la*(jz.dot(jz)-ua.dot(jz)**2)+.01
                hxz=IOR/lw*(jx.dot(jz)-uw.dot(jx)*uw.dot(jz))+1/la*(jx.dot(jz)-ua.dot(jx)*ua.dot(jz))
                det=ti.max(.0001,hxx*hzz-hxz*hxz)
                step=ti.Vector([hzz*gx-hxz*gz,hxx*gz-hxz*gx])/det
                step*=ti.min(1.,.35/ti.max(.00001,step.norm()))
                uv-=step
            surface=self.sea.sample(uv.x,uv.y,clock)
            q=ti.Vector([uv.x,surface.x,uv.y])
            water_ray=(q-water).normalized()
            air_ray=(air-q).normalized()
            normal=ti.Vector([-surface.y,1.,-surface.z]).normalized()
            cosine=water_ray.dot(normal)
            tangent_water=water_ray-normal*cosine
            tangent_air=air_ray-normal*air_ray.dot(normal)
            residual=(IOR*tangent_water-tangent_air).norm()
            water_length=(q-water).norm()
            # Reject non-converged interface paths.
            if residual>.025 or cosine<=0:
                visible=0.
            ray=(q-eye).normalized()
        # Solid intersection calls remain outside the interface solver's
        # dynamic branch; the two segments share this single call site.
        visibility=1.
        for segment in range(2):
            a,b=eye,position
            if eye_wet != point_wet:
                a,b=eye,q
                if segment==1: a,b=q,position
            clear=self.clear_segment(a,b)
            visibility*=clear
        return ray,water_length,visible*visibility

    @ti.func
    def drop_visible(self,eye,position,clock):
        _,_,visible=self.optical_particle(eye,position,clock)
        return visible

    @ti.kernel
    def _prepare_drops(self,eye:ti.types.vector(3,ti.f32),clock:ti.f32):
        for i in range(DROP_CAP):
            self.drop_visibility[i]=0.
            if self.drop_alive[i]==1 and self.drop_aux[i].x>=0:
                ray,water_length,visible=self.optical_particle(eye,self.drop_pos[i],clock)
                self.drop_path[i]=ti.Vector([ray.x,ray.y,ray.z,water_length])
                self.drop_visibility[i]=visible

    @ti.kernel
    def _splat_drops(self,eye:ti.types.vector(3,ti.f32),forward:ti.types.vector(3,ti.f32),
                     right:ti.types.vector(3,ti.f32),up:ti.types.vector(3,ti.f32),
                     scale:ti.f32,clock:ti.f32,lighting:ti.f32,clarity:ti.f32):
        sun=self.sun_direction(lighting)
        for i in range(DROP_CAP):
            if self.drop_alive[i]==1 and self.drop_visibility[i]>.5:
                position,aux=self.drop_pos[i],self.drop_aux[i]
                ray=self.drop_path[i].xyz
                cz=ray.dot(forward)
                distance=(position-eye).norm()
                view=-ray
                reflected=(2*view.dot(self.drop_normal[i])*self.drop_normal[i]-view).normalized()
                tint=self.environment.sample_map(self.environment.environment,reflected,lighting)
                fade=1-self.smooth(aux.y-.35,aux.y,aux.x)
                if self.drop_kind[i]==2:
                    fade=1-self.smooth(.36,.62,aux.x)
                if cz>.04 and distance<120 and fade>0:
                    px=.5*self.width+ray.dot(right)/cz*.5*self.height/scale
                    py=.5*self.height+ray.dot(up)/cz*.5*self.height/scale
                    radius=ti.min(24.,aux.z*.5*self.height/(scale*ti.max(.1,distance*cz)))
                    reach=ti.cast(ti.ceil(ti.max(1.,radius*1.6)),ti.i32)
                    center_x,center_y=ti.cast(ti.floor(px),ti.i32),ti.cast(ti.floor(py),ti.i32)
                    for gx,gy in ti.ndrange((-reach,reach+1),(-reach,reach+1)):
                        x,y=center_x+gx,center_y+gy
                        if 0<=x<self.width and 0<=y<self.height:
                            coverage=0.
                            for sample in ti.static(range(4)):
                                if sample<self.particle_samples[None]:
                                    dx,dy=.5,.5
                                    if self.particle_samples[None]==4:
                                        dx,dy=.25+.5*(sample%2),.25+.5*(sample//2)
                                    d2=((x+dx-px)**2+(y+dy-py)**2)/ti.max(.16,radius*radius)
                                    coverage+=ti.exp(-2*d2)/self.particle_samples[None]
                            alpha=ti.min(.7,radius*.45)*fade*coverage
                            if self.drop_kind[i]==2:
                                alpha*=.65
                            if self.drop_kind[i]==1:
                                alpha*=.28
                            nx,ny=(x+.5-px)/ti.max(.5,radius),(y+.5-py)/ti.max(.5,radius)
                            nz=ti.sqrt(ti.max(.02,1-ti.min(.98,nx*nx+ny*ny)))
                            n=(view*nz+right*nx+up*ny).normalized()
                            if self.drop_kind[i]==2:
                                sheet=self.drop_normal[i]
                                if sheet.dot(view)<0: sheet=-sheet
                                n=(sheet*.65+n*.35).normalized()
                            reflection=(2*view.dot(n)*n-view).normalized()
                            fresnel=.02+.98*ti.pow(1-ti.max(0.,n.dot(view)),5)
                            glint=ti.pow(ti.max(0.,reflection.dot(sun)),180)*3.5
                            rx=ti.max(0,ti.min(self.width-1,x+ti.cast(nx*radius*.25,ti.i32)))
                            ry=ti.max(0,ti.min(self.height-1,y+ti.cast(ny*radius*.25,ti.i32)))
                            transmitted=self.post.hdr[rx,ry]*ti.Vector([.94,.985,1.])
                            color=transmitted*(1-fresnel)+tint*fresnel+ti.Vector([1.,.93,.8])*glint
                            if self.drop_kind[i]==2:
                                # Thin aerated crest, with a transparent body below.
                                band=self.drop_origin[i].w
                                crest=self.smooth(.70,1.,band)
                                lobes=.65+.35*ti.sin(9*aux.w+1.2)
                                froth=.42*crest*lobes*ti.min(1.,aux.x/.08)*self.splash_controls[None].y
                                color=color*(1-froth)+ti.Vector([.72,.86,.94])*(.8+.35*lighting)*froth
                            color=self.water_body(color,self.drop_path[i].w,clarity,0.,n,lighting)
                            ti.atomic_add(self.splash_accum[x,y],color*alpha)
                            ti.atomic_add(self.splash_weight[x,y],alpha)
                    # A faint projected velocity streak accompanies coarse drops.
                    if self.drop_kind[i]!=2 and self.medium_flag[None]<.5:
                        previous=self.drop_prev[i]-eye
                        depth=previous.dot(forward)
                        if depth>.1:
                            tx=.5*self.width+previous.dot(right)/depth*.5*self.height/scale
                            ty=.5*self.height+previous.dot(up)/depth*.5*self.height/scale
                            for k in range(3):
                                fraction=(k+1)/4.
                                x=ti.cast(tx+(px-tx)*fraction,ti.i32)
                                y=ti.cast(ty+(py-ty)*fraction,ti.i32)
                                if 0<=x<self.width and 0<=y<self.height:
                                    alpha=.05*fade*self.splash_controls[None].z*fraction
                                    color=self.post.hdr[x,y]*.9+tint*.1
                                    ti.atomic_add(self.splash_accum[x,y],color*alpha)
                                    ti.atomic_add(self.splash_weight[x,y],alpha)

    @ti.kernel
    def _composite_splash(self,strength:ti.f32):
        for x,y in self.image:
            weight=self.splash_weight[x,y]
            if weight>1e-8:
                alpha=1-ti.exp(-weight*strength)
                color=self.splash_accum[x,y]/weight
                self.post.hdr[x,y]=self.post.hdr[x,y]*(1-alpha)+color*alpha
            self.splash_accum[x,y]=ti.Vector([0.,0.,0.])
            self.splash_weight[x,y]=0.

    def splash(self,x,z,clock,energy=1.,drops=180):
        if not all(math.isfinite(v) for v in (x,z,clock,energy)) or energy<=0 or not 0<=drops<=DROP_CAP-CROWN_CAP:
            raise ValueError("splash requires finite coordinates, positive energy, and available particle capacity")
        if abs(x)>SEA_HALF-3 or abs(z)>SEA_HALF-3 or self.surface_probe(x,z,clock)-bed_height_py(x,z)<.03:
            return False
        self.pulse(float(x),float(z),.20*min(1.6,energy))
        self._foam_burst(float(x),float(z),.65*min(1.6,energy)*float(self.splash_controls[None].y))
        self.sea._sync_view()
        self._burst_drops(float(x),float(z),clock,int(drops),float(energy))
        self._burst_crown(float(x),float(z),clock,float(energy))
        return True

    def step_drops(self,steps,clock):
        for _ in range(int(steps)):
            self._advect_drops(SIM_DT,clock)
            self._landed_ripples()
            self._apply_landing_foam()
            self.sea._sync_view()

    @ti.kernel
    def _pick_surface(self,eye:ti.types.vector(3,ti.f32),ray:ti.types.vector(3,ti.f32),clock:ti.f32)->ti.types.vector(4,ti.f32):
        limit=self.occlusion_distance(eye,ray,120.)
        t=0.001
        point=eye+ray*t
        error=point.y-self.wave_height(point.x,point.z,clock)
        hit=ti.Vector([0.,0.,0.,0.])
        count=0
        while t<limit and count<512 and hit.w==0:
            step=ti.max(.003,ti.min(.5,.8*ti.abs(error)/(ti.abs(ray.y)+2.3*(ti.abs(ray.x)+ti.abs(ray.z))+.001)))
            next_t=ti.min(limit,t+step)
            q=eye+ray*next_t
            new=q.y-self.wave_height(q.x,q.z,clock)
            if error*new<=0:
                lo,hi=t,next_t
                for _ in range(10):
                    mid=(lo+hi)*.5
                    p=eye+ray*mid
                    value=p.y-self.wave_height(p.x,p.z,clock)
                    if error*value>0: lo=mid
                    else: hi=mid
                p=eye+ray*((lo+hi)*.5)
                if self.sea.sample(p.x,p.z,clock).w>.03:
                    hit=ti.Vector([p.x,p.y,p.z,1.])
                t=limit
            else:
                t,error=next_t,new
            count+=1
        return hit

    def pick_water(self,camera,u,v):
        eye,forward,right,up=camera.basis()
        scale=math.tan(math.radians(camera.fov/2))
        ray=forward+right*(2*u-1)*self.width/self.height*scale+up*(2*v-1)*scale
        hit=self._pick_surface(eye,ray/np.linalg.norm(ray),self.sea.time)
        return (float(hit.x),float(hit.z)) if hit.w>.5 else None

    def prepare(self,camera,clock,clarity=1.4,lighting=1.,exposure=1.05):
        super().prepare(camera,clock,clarity,lighting,exposure)
        eye,forward,_,_=camera.basis()
        self._pick_surface(eye,forward,clock)

    def update_stone(self,dt,clock):
        """Fixed-step falling stone, followed by a short submerged descent."""
        if self.stone_pos is None:
            self.stone_wait-=dt
            if self.stone_wait<=0:
                x,z=float(self.rng.uniform(-22.8,-21.2)),float(self.rng.uniform(-.8,.8))
                self.stone_pos=np.array([x,5.,z]);self.stone_vy=0.
                self.stone_sinking=False;self.stone_sink_age=0.
                self.box_extent[self.stone_index]=[.26,.20,.22]
                self.box_bevel[self.stone_index]=.06
                self.box_center[self.stone_index]=self.stone_pos
            return None
        if self.stone_sinking:
            self.stone_sink_age+=dt
            self.stone_vy*=math.exp(-5*dt)
            self.stone_pos[1]=max(bed_height_py(self.stone_pos[0],self.stone_pos[2])+.20,self.stone_pos[1]+self.stone_vy*dt)
            if self.stone_sink_age>=.65:
                self.stone_pos=None;self.stone_wait=2.4
                self.box_center[self.stone_index]=[0.,-60.,0.]
                self.box_extent[self.stone_index]=[.001,.001,.001]
            else:
                self.box_center[self.stone_index]=self.stone_pos
            return None
        previous=self.stone_pos[1]
        self.stone_pos[1]+=self.stone_vy*dt-.5*GRAVITY*dt*dt
        self.stone_vy-=GRAVITY*dt
        surface=float(self.surface_probe(float(self.stone_pos[0]),float(self.stone_pos[2]),clock))
        impact=None
        if self.stone_pos[1]<=surface+.20:
            self.stone_pos[1]=surface+.20
            self.stone_sinking=True
            impact=(float(self.stone_pos[0]),float(self.stone_pos[2]),float(-self.stone_vy))
        self.box_center[self.stone_index]=self.stone_pos
        return impact

    def advance(self,steps,auto_stone=True):
        for _ in range(int(steps)):
            self.sea.advance(1)
            clock=self.sea.time
            self.step_drops(1,clock)
            if auto_stone:
                impact=self.update_stone(SIM_DT,clock)
                if impact is not None:
                    self.splash(impact[0],impact[1],clock,min(1.5,impact[2]/7.),180)
        return self.sea.time

    @ti.func
    def detail_gradient(self,x,z,clock,footprint):
        return DiveRenderer.detail_gradient(self,x,z,clock,footprint)*.35

    @ti.func
    def bed_ripple_strength(self):
        # Keep faint sediment relief without dominating the splash scene.
        return .12

    def composite_effects(self,eye,forward,right,up,scale,clock,lighting,clarity):
        self.particle_samples[None]=self.samples
        self._prepare_drops(eye,clock)
        self._splat_drops(eye,forward,right,up,scale,clock,lighting,clarity)
        self._composite_splash(float(self.splash_controls[None].x))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 08: splashing stones and spray")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, choices=(1, 4), default=4, help="Spatial samples per pixel (default: 4)")
    parser.add_argument("--lighting", choices=("day", "sunset"), default="day",
                        help="Day gives the sun elevation the glints need")
    parser.add_argument("--shadow-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--water-roughness", type=float, default=0.10)
    parser.add_argument("--reflection-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-water-detail", action="store_true")
    parser.add_argument("--no-bloom", action="store_true")
    parser.add_argument("--sea", choices=("auto", "calm", "surf", "storm"), default="auto",
                        help="Sea preset; auto uses the amplitude/period values below")
    parser.add_argument("--wave-amplitude", type=float, default=0.18, help="Incident wave amplitude in metres")
    parser.add_argument("--wave-period", type=float, default=10.0, help="Incident wave period in seconds")
    parser.add_argument("--preset", choices=tuple(SPLASH_PRESETS), default="impact")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_08.png"))
    parser.add_argument("--time", type=float, default=20.0, help="Simulation warmup in seconds before the first frame")
    parser.add_argument("--frames", type=int, default=1, help="Headless frames at a fixed 1/60 s step; save the last")
    parser.add_argument("--window-frames", type=int, default=0, help="Close interactive window after N frames (0: unlimited)")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk compilation cache for troubleshooting")
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.frames < 1 or args.window_frames < 0 or not math.isfinite(args.time):
        parser.error("width/height must be >= 32, frames >= 1, and time finite")
    if not math.isfinite(args.water_roughness) or not 0.0 <= args.water_roughness <= 0.45:
        parser.error("water-roughness must be finite and between 0 and 0.45")
    if not math.isfinite(args.wave_amplitude) or not 0.05 <= args.wave_amplitude <= 0.6:
        parser.error("wave-amplitude must be finite and between 0.05 and 0.6")
    if not math.isfinite(args.wave_period) or not 6.0 <= args.wave_period <= 16.0:
        parser.error("wave-period must be finite and between 6 and 16")
    return args


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = SplashRenderer(args.width, args.height, args.samples if args.headless else 4)
    renderer.samples = args.samples
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 1.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = DiveCamera()
    apply_splash_preset(camera, args.preset)
    amplitude, period = sea_parameters(args)
    renderer.sea.set_wave(amplitude, period)
    wave = [amplitude, period]
    renderer._wave_key = (amplitude, period)
    speed = 1.0
    lighting = 0.0 if args.lighting == "sunset" else 1.0
    clarity, exposure = 1.4, 1.05
    clock = 0.0
    sim_accumulator = 0.0

    def advance(substeps):
        nonlocal clock
        if substeps>0:
            clock=renderer.advance(substeps)

    advance(int(max(0.,args.time)/SIM_DT))
    # Start on the readable expansion stage; subsequent drops remain cyclic.
    renderer.reset_drops()
    renderer.splash(-22.,0.,clock,energy=1.2,drops=180)
    clock=renderer.advance(5,auto_stone=False)
    renderer.stone_wait=2.0

    if args.headless:
        start = time.perf_counter()
        timings = []
        for frame in range(args.frames):
            frame_start = time.perf_counter()
            frame_steps, sim_accumulator = simulation_budget(sim_accumulator, 1.0 / 60.0, 1.0)
            advance(frame_steps)
            renderer.draw(camera, clock, clarity, lighting, exposure)
            ti.sync()
            timings.append(time.perf_counter() - frame_start)
        print(f"Saved {renderer.save(args.output)} ({args.width}x{args.height})")
        print(f"{args.frames} frame(s), including compilation: {time.perf_counter() - start:.2f}s")
        if len(timings) > 1:
            print(f"Post-compilation render mean: {np.mean(timings[1:]) * 1000:.1f} ms/frame")
        return

    # Cold JIT compilation must finish before creating a native window: GGUI
    # polls OS events in show(), which cannot run while draw() is compiling.
    print("[Startup] Preparing first frame; cold compilation can take a minute or more. Window opens when ready.", flush=True)
    warmup_start = time.perf_counter()
    renderer.prepare(camera, clock, clarity, lighting, exposure)
    ti.sync()
    print(f"[Startup] First frame ready in {time.perf_counter() - warmup_start:.2f}s. Opening window...", flush=True)
    window = ti.ui.Window("08 / Splash", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    previous_time, previous_mouse = time.perf_counter(), None
    sim_accumulator, frame_ms, window_frames = 0.0, 0.0, 1
    drag_in_scene, click_start, click_camera_moved = False, None, False
    preset_keys = {"1": "overview", "2": "waterline", "3": "top", "4": "dive",
                   "5": "window", "6": "bed", "7": "impact"}
    print("Click water: splash | Drag LMB orbit | RMB pan | W/S zoom | 1-7 views (7 impact) | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
    while window.running and (not args.window_frames or window_frames < args.window_frames):
        now = time.perf_counter()
        dt = min(now - previous_time, 0.25)
        previous_time = now
        step, save_frame = False, False
        mouse = np.array(window.get_cursor_pos())
        in_panel = show_panel and ((mouse[0] < 0.33 and mouse[1] > 0.15) or (mouse[0] > 0.70 and mouse[1] > 0.04))
        for event in window.get_events(ti.ui.PRESS):
            key = event.key
            if key == ti.ui.ESCAPE:
                window.running = False
            elif key == " ":
                paused = not paused
            elif key == "n":
                step = True
            elif key == "a":
                auto_orbit = not auto_orbit
            elif key == "h":
                show_panel = not show_panel
            elif key in preset_keys:
                apply_splash_preset(camera, preset_keys[key])
                auto_orbit = False
            elif key == "r":
                apply_splash_preset(camera, "impact")
                renderer.samples = args.samples
                renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
                renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 1.0, args.reflection_samples]
                renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
                renderer.splash_controls[None] = [1.0, 1.0, 0.6, 0.0]
                amplitude, period = sea_parameters(args)
                wave = [amplitude, period]
                renderer.sea.reset()
                renderer.sea.set_wave(amplitude, period)
                renderer._wave_key = (amplitude, period)
                renderer.reset_drops()
                lighting = 0.0 if args.lighting == "sunset" else 1.0
                clarity, exposure, clock = 1.4, 1.05, 0.0
                sim_accumulator = 0.0
                paused, auto_orbit = False, False
            elif key == "p":
                save_frame = True
            elif key in (ti.ui.LMB, ti.ui.RMB):
                drag_in_scene = not in_panel
        if not window.running:
            break
        # A short click splashes; a longer drag keeps orbiting.
        if drag_in_scene and window.is_pressed(ti.ui.LMB):
            if click_start is None:
                click_start, click_camera_moved = mouse.copy(), False
            elif not click_camera_moved and np.linalg.norm(mouse - click_start) > 0.012:
                click_camera_moved = True
        elif click_start is not None:
            if drag_in_scene and not click_camera_moved:
                point = renderer.pick_water(camera, click_start[0], click_start[1])
                if point is not None:
                    renderer.splash(point[0], point[1], clock, energy=0.85, drops=70)
            click_start = None
        if previous_mouse is not None and drag_in_scene and (click_camera_moved or window.is_pressed(ti.ui.RMB)):
            delta = mouse - previous_mouse
            if window.is_pressed(ti.ui.LMB):
                camera.orbit(*delta)
                auto_orbit = False
            elif window.is_pressed(ti.ui.RMB):
                camera.pan(*delta)
                auto_orbit = False
        previous_mouse = mouse
        if window.is_pressed("w", ti.ui.UP):
            camera.zoom(-dt)
        if window.is_pressed("s", ti.ui.DOWN):
            camera.zoom(dt)
        if auto_orbit:
            camera.yaw += dt * 0.12
            camera.lift_above_bed()
        if show_panel:
            with gui.sub_window("LIGHT / MATERIALS", 0.72, 0.025, 0.26, 0.28):
                controls = renderer.light_controls[None]
                controls[1] = 4 if gui.checkbox("Soft sun shadows", controls[1] == 4) else 1
                controls[0] = math.radians(gui.slider_float("Sun radius (deg)", math.degrees(controls[0]), 0.0, 3.0))
                controls[2] = gui.slider_float("Contact AO", controls[2], 0.0, 1.0)
                controls[3] = gui.slider_float("Environment", controls[3], 0.0, 2.0)
                renderer.light_controls[None] = controls
            with gui.sub_window("SHORE / FINISH", 0.72, 0.335, 0.26, 0.52):
                controls = renderer.water_controls[None]
                controls[0] = gui.slider_float("Water roughness", controls[0], 0.0, 0.45)
                controls[1] = gui.slider_float("Fine waves", controls[1], 0.0, 2.0)
                controls[2] = gui.slider_float("Foam amount", controls[2], 0.0, 2.0)
                controls[3] = 4 if gui.checkbox("4x reflections", controls[3] == 4) else 1
                renderer.water_controls[None] = controls
                dive = renderer.dive_controls[None]
                dive[0] = gui.slider_float("Caustics", dive[0], 0.0, 2.0)
                dive[1] = gui.slider_float("Sun shafts", dive[1], 0.0, 2.0)
                renderer.dive_controls[None] = dive
                splash = renderer.splash_controls[None]
                splash[0] = gui.slider_float("Spray brightness", splash[0], 0.0, 2.0)
                splash[1] = gui.slider_float("Splash foam", splash[1], 0.0, 2.0)
                splash[2] = gui.slider_float("Spray trails", splash[2], 0.0, 1.0)
                renderer.splash_controls[None] = splash
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
            with gui.sub_window("SPLASH / 08", 0.02, 0.025, 0.30, 0.78):
                gui.text("Stones fall and burst into spray")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
                if gui.button("Single step [N]"):
                    step = True
                if gui.button("Drop a stone now"):
                    renderer.stone_wait = 0.0
                wave[0] = gui.slider_float("Wave amplitude", wave[0], 0.05, 0.6)
                wave[1] = gui.slider_float("Wave period", wave[1], 6.0, 16.0)
                speed = gui.slider_float("Time speed", speed, 0.0, 2.0)
                clarity = gui.slider_float("Water clarity", clarity, 0.35, 3.0)
                lighting = gui.slider_float("Day / sunset", lighting, 0.0, 1.0)
                exposure = gui.slider_float("Exposure", exposure, 0.5, 1.8)
                renderer.samples = 4 if gui.checkbox("4x spatial AA", renderer.samples == 4) else 1
                if gui.button("Impact view [7]"):
                    apply_splash_preset(camera, "impact")
                    auto_orbit = False
                if gui.button("Overview [1]"):
                    apply_splash_preset(camera, "overview")
                    auto_orbit = False
                if gui.button("Dive [4]"):
                    apply_splash_preset(camera, "dive")
                    auto_orbit = False
                if gui.button("Snell window [5]"):
                    apply_splash_preset(camera, "window")
                    auto_orbit = False
                gui.text("Click water for a splash")
                gui.text("Drag down to submerge the camera")
                gui.text("Sea presets: --sea calm/surf/storm")
                gui.text("W/S or arrows: zoom / R: reset")
                gui.text("H: hide panel / P: save / Esc: exit")
        # The incident wave follows the sliders on the host.
        current = (float(wave[0]), float(wave[1]))
        if current != renderer._wave_key:
            renderer.sea.set_wave(*current)
            renderer._wave_key = current
        if paused:
            if step:
                advance(1)
        else:
            substeps, sim_accumulator = simulation_budget(sim_accumulator, dt, speed)
            advance(substeps)
        renderer.draw(camera, clock, clarity, lighting, exposure)
        if save_frame:
            print(f"Saved {renderer.save(args.output)}")
        canvas.set_image(renderer.image)
        window.show()
        window_frames += 1
        if args.window_frames and window_frames >= args.window_frames:
            window.running = False
        frame_ms = 0.9 * frame_ms + 0.1 * (time.perf_counter() - now) * 1000


if __name__ == "__main__":
    main()
