# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from abc import ABC, abstractmethod

class HandlerFactory:
  # this will be populated at "compile time"
  # by the get_handler decorator
  # When the decorators are executed, this dict will look something like…
  file_handlers = {}

  @classmethod
  def register_handler(cls, output_type):
    def wrapper(handler_cls):
      cls.file_handlers[output_type] = handler_cls
      return handler_cls
    return wrapper

  @classmethod
  def get_handler(cls, output_type):
    try:
        return cls.file_handlers[output_type]()
    except KeyError:
        raise NotImplementedError(f"'{output_type}' is not supported")

class FileHandler(ABC):
  @abstractmethod
  def write(self, path):
    raise NotImplementedError

# Any classes with this decorator will automatically
# register itself with the Factory dict
@HandlerFactory.register_handler("csv")
class CSVHandler(FileHandler):
  def write(self, data):
    # Some implementation to write data out as CSV
    print("Successfully wrote to CSV!")

@HandlerFactory.register_handler("json")
class JSONHandler(FileHandler):
  def write(self, data):
    # Some implementation to write data out as JSON
    print("Successfully wrote to JSON!")
    
@HandlerFactory.register_handler("excel")
class ExcelHandler(FileHandler):
  def write(self, data):
    # Some implementation to write data out as JSON
    print("Successfully wrote to xlsx!")

handler = HandlerFactory.get_handler("txt")
handler.write("dummy-data")


#%%

def fns():
    
    handlers = {}
    def register_handler(kind):
      def wrapper(fn):
        handlers[kind] = fn
        return fn
      return wrapper
    
    @register_handler('a')
    def fn_a():
        print("ran fn with kind = a")
        
    @register_handler('b')
    def fn_b():
        print("ran fn with kind = b")
        
    return handlers
        
def fn(kind):
    handlers = fns()
    try:
        return handlers[kind]()
    except KeyError:
        raise NotImplementedError(f"kind ='{kind}' is not supported")
#%%
fn(kind = 'd')
