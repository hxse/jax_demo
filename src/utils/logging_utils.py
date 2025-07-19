import logging
import jax
import re


class CudaErrorFilter(logging.Filter):

    def filter(self, record):
        message = record.getMessage()
        pattern = r"Jax plugin configuration error: Exception when calling jax_plugins\.xla_cuda\d+\.initialize"
        return not re.search(pattern, message)


def configure_cuda_error_logging():
    logger = logging.getLogger("jax._src.xla_bridge")
    logger.addFilter(CudaErrorFilter())
