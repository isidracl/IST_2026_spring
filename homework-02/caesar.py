import typing as tp


def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """
    ciphertext = ""
    for char in plaintext:
        if char.isalpha():
            if char.isupper():
                shifted = (ord(char) - ord('A') + shift) % 26
                ciphertext += chr(ord('A') + shifted)
            else:
                shifted = (ord(char) - ord('a') + shift) % 26
                ciphertext += chr(ord('a') + shifted)
        else:
            ciphertext += char
    return ciphertext
    return ciphertext


def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    plaintext = ""
    for char in ciphertext:
        if char.isalpha():
            if char.isupper():
                shifted = (ord(char) - ord('A') - shift) % 26
                plaintext += chr(ord('A') + shifted)
            else:
                shifted = (ord(char) - ord('a') - shift) % 26
                plaintext += chr(ord('a') + shifted)
        else:
            plaintext += char
    return plaintext
    return plaintext


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0
    max_matches = 0
    
    for shift in range(26):
        decrypted = decrypt_caesar(ciphertext, shift)
        words = decrypted.lower().split()
        matches = sum(1 for word in words if word in dictionary)
        
        if matches > max_matches:
            max_matches = matches
            best_shift = shift
    
    return best_shift
    return best_shift
