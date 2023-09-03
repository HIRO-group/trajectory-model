import struct

def read_vectors(filename):
    vectors = []
    with open(filename, 'rb') as f:
        num_vectors = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_vectors):
            vector = [struct.unpack('d', f.read(8))[0] for _ in range(7)] 
            vectors.append(vector)
            
    return vectors
