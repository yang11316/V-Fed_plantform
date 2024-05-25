from hashlib import md5


def MD5_hashCode(string):

    plainTextBytes = string.encode('utf-8')
    encryptor = md5()
    encryptor.update(plainTextBytes)
    hashCode = encryptor.hexdigest()
    return hashCode


if __name__=="__main__":
    print(MD5_hashCode('123456@123.com'))