import sys
import json
import uuid
import argparse
import uvicorn

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from tts import TTS

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    request_id: str
    code: str
    error: str


# Global variables
timeout_keep_alive = 5  # seconds

# Setting up Translator 
tts = TTS()


# Exception Handler
async def unhandledExceptionHandler(request: Request, exc: Exception) -> JSONResponse:
    """
    This middleware will log all unhandled exceptions.
    Unhandled exceptions are all exceptions that are not HTTPExceptions or RequestValidationErrors.
    """
    id = str(uuid.uuid4())
    exception_type, exception_value, exception_traceback = sys.exc_info()
    exception_name = getattr(exception_type, "__name__", None)
    response = ErrorResponse(request_id=id, code=str(500000), error=str(exception_name)).model_dump()
    return JSONResponse(response, status_code=500)


# Create FastAPI app
app = FastAPI(title="G11n Audio IQ Service",
    summary="Vector Cache service to store and search embeddings",
    version="0.1",
    openapi_tags=[
        {
            "name": "basic",
            "description": "Common API(s)",
        },
        {
            "name": "v1",
            "description": "Version 1 API(s)"
        },
    ],
    debug=True
)
app.add_exception_handler(Exception, unhandledExceptionHandler)


# Health Check API
@app.get('/health')
async def health() -> Response:
    return JSONResponse("OK", status_code=200)


# Translate API
@app.post('/v1/audio')
async def translate(request: Request) -> Response:
    # Reading input request data
    request_dict = await request.json()
    if 'request_id' in request_dict:
        id = str(request_dict.pop("request_id"))
    else:
        id = str(uuid.uuid4())
        
    if 'text' in request_dict:
        text = request_dict.pop("text")
        if isinstance(text, list):
            text = list(map(str, text))
        else:
            text = str(text)
    else:
        ret = ErrorResponse(request_id=id, code=str(422001), error="Required field `text` missing in request").model_dump()
        return JSONResponse(ret, status_code=422)

    if 'locale' in request_dict:
        locale = str(request_dict.pop("locale"))
    else:
        ret = ErrorResponse(request_id=id, code=str(422001), error="Required field `locale` missing in request").model_dump()
        return JSONResponse(ret, status_code=422)

    if 'speaker' in request_dict:
        speaker = str(request_dict.pop("speaker"))
    else:
        speaker = None
        
    try:
        audio = tts.generate_audio(text=text, locale=locale, speaker=speaker)
        tts.save_audio(audio, filename=f"{id}.wav")
        ret = {
            "request_id": id,
            "response": f"{id}.wav"
        }
        return JSONResponse(ret)
        
    except Exception as e:
        ret = ErrorResponse(request_id=id, code=str(500), error=str(e)).model_dump()
        return JSONResponse(ret, status_code=500)
    

# Run Application
if __name__ == "__main__":
    import argparse
    
    # Setting configurable parameters
    parser = argparse.ArgumentParser(description="RESTful API server.")

    # Uvicorn parameters
    parser.add_argument("--host", type=str, default='0.0.0.0', help="Hostname")
    parser.add_argument("--port", type=int, default=6006, help="Port")

    # Fastapi parameters
    parser.add_argument("--allow-credentials",
        action="store_true",
        default=True,
        help="allow credentials")
    parser.add_argument("--allowed-origins",
        type=json.loads,
        default=["*"],
        help="allowed origins")
    parser.add_argument("--allowed-methods",
        type=json.loads,
        default=["*"],
        help="allowed methods")
    parser.add_argument("--allowed-headers",
        type=json.loads,
        default=["*"],
        help="allowed headers")
    
    
    args = parser.parse_args()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers
    )

        
    # Uvicorn configuration
    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "loggers": {
            "uvicorn": {"level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"level": "INFO", "propagate": False},
        },
    }
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=timeout_keep_alive, log_config=uvicorn_log_config)