class NvidiaAPIError(Exception):
    pass


class MissingCredentialError(NvidiaAPIError):
    pass


class RequestFailedError(NvidiaAPIError):
    def __init__(self, message: str, status_code: int | None = None, response_text: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
