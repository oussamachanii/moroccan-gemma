import http.server
import socketserver

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        print(f"Received request: {self.path}")
        # Return a basic JSON response
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = '{"documents": [["Context for Moroccan dialect: Darija is the primary spoken language in Morocco."]]}'
        self.wfile.write(response.encode("utf-8"))

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving mock ChromaDB API at port {PORT}")
    httpd.serve_forever()