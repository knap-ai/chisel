import os

class EnvHandler:
    """
    A class for getting, setting and checking environment variables.
    """

    @staticmethod
    def get(key, default=None):
        """
        Get the value of an environment variable.
        """
        return os.environ.get(key, default)

    @staticmethod
    def set(key, value):
        """
        Set the value of an environment variable.
        """
        os.environ[key] = value

    @staticmethod
    def contains(key):
        """
        Check if an environment variable is set.
        """
        return key in os.environ
