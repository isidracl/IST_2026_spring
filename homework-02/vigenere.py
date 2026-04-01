def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    """
    Encrypts plaintext using a Vigenere cipher.

    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    ciphertext = ""
    keyword_index = 0
    
    for char in plaintext:
        if char.isalpha():
            key_char = keyword[keyword_index % len(keyword)]
            shift = ord(key_char.upper()) - ord('A')
            if char.isupper():
                shifted = (ord(char) - ord('A') + shift) % 26
                ciphertext += chr(ord('A') + shifted)
            else:
                shifted = (ord(char) - ord('a') + shift) % 26
                ciphertext += chr(ord('a') + shifted)
        else:
            ciphertext += char
        keyword_index += 1
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    keyword_index = 0
    
    for char in ciphertext:
      key_char = keyword[keyword_index % len(keyword)]
      shift = ord(key_char.upper()) - ord('A')

      if char.isalpha():
        if char.isupper():
          shifted = (ord(char) - ord('A') - shift) % 26
          plaintext += chr(ord('A') + shifted)
        else:
          shifted = (ord(char) - ord('a') - shift) % 26
          plaintext += chr(ord('a') + shifted)
      else:
        plaintext += char
      keyword_index += 1
    return plaintext
