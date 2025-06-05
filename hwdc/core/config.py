from distutils.util import strtobool
from os import environ

HWDC_DEBUG: bool = bool(strtobool(environ.get('HWDC_DEBUG', 'false')))
