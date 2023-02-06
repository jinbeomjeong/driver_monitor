import socket


class TcpServerCom:
    def __init__(self, addr='localhost', port=6340):
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.server_socket.bind((addr, port))
        self.server_socket.listen(1)
        print("client listening!")

        self.connection_socket, client_addr = self.server_socket.accept()
        print('connection form'+str(client_addr[0])+'!')

    def disconnect(self):
        self.server_socket.close()
        print("disconnection successful")

    def receive_msg(self):
        output = self.connection_socket.recv(1024).decode('utf-8')

        return output

    def send_msg(self, message):
        self.connection_socket.send(message.encode())


class TCPClientCom:
    def __init__(self, addr='localhost', port=6340):
        self.client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.client_socket.connect((addr, port))
        print("TCP connected!")

    def disconnect(self):
        self.client_socket.close()
        print("disconnection successful")

    def send_msg(self, message):
        print(message)
        self.client_socket.sendto(message.encode('utf-8'), (self.local_addr, self.port))

    def receive_msg(self):
        output = self.client_socket.recv(1024).decode('utf-8')

        return output
