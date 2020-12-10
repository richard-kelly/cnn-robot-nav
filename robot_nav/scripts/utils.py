import math


def cart_to_pol(x, y):
    rho = math.sqrt(x**2 + y**2)
    phi = math.atan2(y, x)
    return rho, phi


def pol_to_cart(rho, phi):
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return x, y
