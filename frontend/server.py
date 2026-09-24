from http.server import HTTPServer, SimpleHTTPRequestHandler


PORT = 5500

server = HTTPServer(
    ("127.0.0.1", PORT),
    SimpleHTTPRequestHandler
)

print(f"Frontend running at http://127.0.0.1:{PORT}")

server.serve_forever()