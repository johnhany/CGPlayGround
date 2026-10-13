# 带禁缓存头的静态文件服务器：开发期模块频繁改动，
# python http.server 默认只发 Last-Modified，浏览器启发式缓存
# 会拿到旧模块，造成新旧代码混用。本服务器强制每次回源验证。
import functools
import http.server
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'   # 保持连接，避免浏览器侧连接重置

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # 静默访问日志


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 38517
    handler = functools.partial(NoCacheHandler, directory='.')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
    print(f'静态服务器已启动: http://127.0.0.1:{port} (禁用缓存)')
    server.serve_forever()


if __name__ == '__main__':
    main()
