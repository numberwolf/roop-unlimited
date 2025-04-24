import inspect
def print_info():
    frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(frame)[1]
    file_name = caller_frame.filename
    function_name = caller_frame.function
    line_number = caller_frame.lineno
    print(f"FILENAME: {file_name}, FUNCNAME: {function_name}, LINE: {line_number}")
