
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

path = "C:\\data"  # 设置路径
global Created


def on_created(event):
    global Created
    # print(f"{event.src_path}被创建")
    Created = event.src_path
    return Created


def on_deleted(event):
    print(f"{event.src_path}被删除")
    pass


def on_modified(event):
    print(f"{event.src_path} 被修改")
    pass


def on_moved(event):
    print(f"{event.src_path}被移动到{event.dest_path}")
    pass


event_handler = PatternMatchingEventHandler(patterns=None,
                                            ignore_patterns=None,
                                            ignore_directories=False,
                                            case_sensitive=False)
event_handler.on_created = on_created
event_handler.on_deleted = on_deleted
event_handler.on_modified = on_modified
event_handler.on_moved = on_moved
observer = Observer()  # 创建观察者对象
# file_handler = MyEventHandler()  # 创建事件处理对象
observer.schedule(event_handler, path, False)  # 向观察者对象绑定事件和目录
# file_handler.on_created(observer)
observer.start()  # 启动