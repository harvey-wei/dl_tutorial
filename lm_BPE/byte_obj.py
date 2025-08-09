b1 = bytes([97])       # b'a'
b2 = bytes([98])       # b'b'
b3 = b1 + b2            # b'ab'

print(b3)              # b'ab'
print(list(b3))        # [97, 98]
