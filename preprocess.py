from config import VIDEO_PATH_TEMPLATE

def get_video_file(key):
    file_template = VIDEO_PATH_TEMPLATE.format(key=key)
    return file_template

