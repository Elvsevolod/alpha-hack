"""Bounded verifier feedback; transport failures cannot erase semantic repairs."""
from .util import redact


class ResolutionFeedback:
    def __init__(self, request_id, saved=None):
        self.request_id = request_id
        saved = saved if saved and saved.get('request_id') == request_id else {}
        self.errors = [redact(str(x))[:1500] for x in saved.get('validation_errors', [])[:1]]
        self.response_error = None

    def record(self, error, *, validated_proposal):
        message = redact(str(error))[:1500]
        if validated_proposal:
            # A fresh verifier result supersedes the previous verifier result.
            # Do not keep stale constraints after a corrected proposal advances.
            self.errors = [message]
            self.response_error = None
        else:
            self.response_error = message

    def payload(self):
        return {'validation_errors': list(self.errors), 'response_error': self.response_error}

    def state(self):
        return {'request_id': self.request_id, **self.payload()}
