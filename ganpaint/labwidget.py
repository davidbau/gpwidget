"""
labwidget by David Bau.

Base class for a lightweight javascript notebook widget framework
that is portable across Google colab and Jupyter notebooks.
No use of requirejs: the design uses all inline javascript.

Defines Model, Widget, Event, and Property, which set up data binding
using the communication channels available in either google colab
environment or jupyter notebook.  New can be defined as in the
following example.

class TextWidget(Widget):
  def __init__(self, value='', width=20):
    super().__init__()
    # databinding is defined using Property objects.
    self.value = Property(value)
    self.width = Property(width)
    self.esc = Event()

  def widget_js(self):
    # Both "model" and "element" objects are defined within the scope
    # where the js is run.  "element" looks for the element with id
    # self.view_id(); if widget_html is overridden, this id should be used.
    return '''
      element.value = model.get('value');
      element.width = model.get('width');
      element.addEventListener('keydown', (e) => {
        if (e.code == 'Enter') {
          model.set('value', element.value);
        }
        if (e.code == 'Esc') {
          model.trigger('esc', 'my event payload')
        }
      });
      model.on('value', (value) => {
        element.value = model.get('value');
      });
      model.on('width', (value) => {
        element.width = model.get('width');
      });
    '''
  def widget_html(self):
    return f'''
    <input id="{self.view_id()}"
       value="{html.escape(str(self.value))}" width="{self.width}">
    '''

User interaction should update the javascript model using
model.set('propname', value); this will propagate to the python
model and notify any registered python listeners.

Communication pattern goes in a V shape:
User entry -> js model set -> python model ->  js model get -> User feedback

TODO: Support jupyterlab also.
"""

import json, html
from inspect import signature

class Model(object):
    '''
    Abstract base class that supports data binding.  Within __init__,
    a model subclass defines databound events and properties using:

       self.evtname = Event()
       self.propname = Property(initval)

    Any Event or Property member can be watched by registering a
    listener with `model.on('propname', callback)`.

    An event can be triggered by `model.evtname.trigger(value)`.
    A property can be read with `model.propname`, and can be set by
    `model.propname = value`; this also triggers notifications.
    In both these cases, any registered listeners will be called
    with the given value.
    '''
    def on(self, name, cb):
        '''
        Registers a listener for named events and properties.
        A space-separated list of names can be provided as `name`.
        '''
        for n in name.split():
            self.prop(n).on(cb)
        return self

    def off(self, name, cb):
        '''
        Unregisters a listener for named events and properties.
        A space-separated list of names can be provided as `name`.
        '''
        for n in name.split():
            self.prop(n).off(cb)
        return self

    def prop(self, name):
        '''
        Returns the underlying Event or Property object for a
        property, rather than its held value.
        '''
        curvalue = super().__getattribute__(name)
        if not isinstance(curvalue, Event):
            raise AttributeError('%s not a property or event but %s' 
                    % (name, str(type(curvalue))))
        return curvalue

    def _initprop(self, name, value):
        '''
        To be overridden in base classes.  Handles initialization of
        a new Event or Property member.
        '''
        return

    def __setattr__(self, name, value):
        '''
        When a member is an Event or Property, then assignment notation
        is delegated to the Event or Property so that notifications
        and reparenting can be handled.  That is, `model.name = value`
        turns into `prop(name).set(value)`.
        '''
        if hasattr(self, name):
            curvalue = super().__getattribute__(name)
            if isinstance(curvalue, Event):
                # Delegte "set" to the underlying Property.
                curvalue.set(value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
            if isinstance(value, Event):
                self._initprop(name, value)

    def __getattribute__(self, name):
        '''
        When a member is a Property, then property getter
        notation is delegated to the peoperty object.
        '''
        curvalue = super().__getattribute__(name)
        if isinstance(curvalue, Property):
            return curvalue.value
        return curvalue

class Widget(Model):

    def __init__(self):
        # In the jupyter case, there can be some delay between js injection
        # and comm creation, so we need to queue some initial messages.
        self._comms = []
        self._queue = []
        # Each call to _repr_html_ creates a unique view instance.
        self._viewcount = 0
        def handle_remote_set(name, value):
            self.prop(name).trigger(value)
        self._recv_from_js(handle_remote_set)

    def widget_html(self):
        return f'<div id="{self.view_id()}"></div>'

    def widget_js(self):
        return ''

    def view_id(self):
        return f"_{id(self)}_{self._viewcount}"

    def _repr_html_(self):
        self._viewcount += 1
        json_data = json.dumps({
                k: v.value for k, v in vars(self).items()
                if isinstance(v, Property)})
        return f"""
        {self.widget_html()}
        <script>
        (function() {{
        {WIDGET_MODEL_JS}
        var model = new Model("{id(self)}", {json_data});
        var element = document.getElementById("{self.view_id()}");
        {self.widget_js()}
        }})();
        </script>
        """

    def _initprop(self, name, value):
        if not hasattr(self, '_viewcount'):
            raise ValueError('base Model __init__ must be called')
        def notify_js(value):
            self._send_to_js(id(self), name, value)
        if isinstance(value, Event):
            value.on(notify_js)

    def _send_to_js(self, *args):
        if self._viewcount > 0:
            if WIDGET_ENV == 'colab':
                colab_output.eval_js(f"""
                (window.send_{id(self)} = window.send_{id(self)} ||
                new BroadcastChannel("channel_{id(self)}")
                ).postMessage({json.dumps(args)});
                """, ignore_result=True)
            elif WIDGET_ENV == 'jupyter':
                if not self._comms:
                    self._queue.append(args)
                    return
                for comm in self._comms:
                    comm.send(args)

    def _recv_from_js(self, fn):
        if WIDGET_ENV == 'colab':
            colab_output.register_callback(f"invoke_{id(self)}", fn)
        elif WIDGET_ENV == 'jupyter':
            def handle_comm(msg):
                fn(*(msg['content']['data']))
                # TODO: handle closing also.
            def handle_close(close_msg):
                comm_id = close_msg['content']['comm_id']
                self._comms = [c for c in self._comms if c.comm_id != comm_id]
            def open_comm(comm, open_msg):
                self._comms.append(comm)
                comm.on_msg(handle_comm)
                comm.on_close(handle_close)
                comm.send('ok')
                if self._queue:
                    for args in self._queue:
                        comm.send(args)
                    self._queue.clear()
                if open_msg['content']['data']:
                    handle_comm(open_msg)
            cname = "comm_" + str(id(self))
            get_ipython().kernel.comm_manager.register_target(cname, open_comm)

class Event(object):
    """
    Event is the base class for Property and other data-bound
    field objects.  Event holds a list of listeners that need to
    be notified about the event.

    Multple Event objects can be tied (typically a parent Model can
    have Events that are triggered by children models).  To support
    this, each Event can have a parent.

    Event objects provide a notification protocol where view
    interactions cause triggers at a leaf that are sent up to the
    root Event to be handled.  By default, the root handler accepts
    triggers by notifying all listeners and children in the tree.
    """
    def __init__(self):
        self._listeners = []
        self.parent = None
    def handle(self, value):
        '''
        Method to override; called at the root when an event has been
        triggered, and on a child when the parent has notified.  By
        default notifies all listeners.
        '''
        self.notify(value)
    def trigger(self, value=None):
        '''
        Triggers an event to be handled by the root.  By default, the root
        handler will accept the event so all the listeners will be notified.
        '''
        if self.parent is not None:
            self.parent.trigger(value)
        else:
            self.handle(value)
    def set(self, value):
        '''
        Sets the parent Event.  Child Events trigger events by
        triggering parents, and in turn they handle notifications
        that come from parents.
        '''
        if self.parent is not None:
            self.parent.off(self.handle)
            self.parent = None
        if isinstance(value, Event):
            ancestor = value.parent
            while ancestor is not None:
                if ancestor == self:
                    raise ValueError('bound properties should not form a loop')
                ancestor = ancestor.parent
            self.parent = value
            self.parent.on(self.handle)
        elif not isinstance(self, Property):
            raise ValueError('only properties can be set to a value')
    def notify(self, value=None):
        '''
        Notifies listeners and children.  If a listener accepts an argument,
        the value will be passed as a single argument.
        '''
        for cb in self._listeners:
            if len(signature(cb).parameters) == 0:
                cb() # no-parameter callback.
            else:
                cb(value)
    def on(self, cb):
        '''
        Registers a listener.  Calling multiple times registers
        multiple listeners.
        '''
        self._listeners.append(cb)
    def off(self, cb):
        '''
        Unregisters a listener.
        '''
        self._listeners = [c for c in self._listeners if c != cb]

class Property(Event):
    """
    A Property is just an Event that remembers its last value.
    """
    def __init__(self, value=None):
        '''
        Can be initialized with a starting value.
        '''
        super().__init__()
        self.set(value)
    def handle(self, value):
        '''
        The default handling for a Property is to store the value,
        then notify listeners.  This method can be overridden,
        for example to validate values.
        '''
        self.value = value
        self.notify(value)
    def set(self, value):
        '''
        When a Property value is set to an ordinary value, it
        triggers an event which causes a notification to be
        sent to update all linked Properties.  A Property set
        to another Property becomes a child of the value.
        '''
        # Handle setting a parent Property
        if isinstance(value, Property):
            super().set(value)
            self.handle(value.value)
        elif isinstance(value, Event):
            raise ValueError('Cannot set a Property to an Event')
        else:
            self.trigger(value)


##########################################################################
## Specific widgets
##########################################################################

class Button(Widget):
    def __init__(self, label='button'):
        super().__init__()
        self.click = Event()
        self.label = Property(label)
    def widget_js(self):
        return '''
          element.addEventListener('click', (e) => {
            model.trigger('click');
          })
          model.on('label', (v) => {
            element.value = v;
          })
        '''
    def widget_html(self):
        return f'''
          <input id="{self.view_id()}" type="button"
            value="{html.escape(str(self.label))}">
        '''

class Label(Widget):
    def __init__(self, value=''):
        super().__init__()
        # databinding is defined using Property objects.
        self.value = Property(value)

    def widget_js(self):
        # Both "model" and "element" objects are defined within the scope
        # where the js is run.    "element" looks for the element with id
        # self.view_id(); if widget_html is overridden, this id should be used.
        return '''
            model.on('value', (value) => {
                element.innerText = model.get('value');
            });
        '''
    def widget_html(self):
        return f'''
        <label id="{self.view_id()}">{html.escape(str(self.value))}</label>
        '''

class Textbox(Widget):
    def __init__(self, value='', size=20):
        super().__init__()
        # databinding is defined using Property objects.
        self.value = Property(value)
        self.size = Property(size)

    def widget_js(self):
        # Both "model" and "element" objects are defined within the scope
        # where the js is run.    "element" looks for the element with id
        # self.view_id(); if widget_html is overridden, this id should be used.
        return '''
          element.value = model.get('value');
          element.size = model.get('size');
          element.addEventListener('keydown', (e) => {
            if (e.code == 'Enter') {
              model.set('value', element.value);
            }
          });
          model.on('value', (value) => {
            element.value = model.get('value');
          });
          model.on('size', (value) => {
            element.size = model.get('size');
          });
        '''
    def widget_html(self):
        return f'''
          <input id="{self.view_id()}"
             value="{html.escape(str(self.value))}" size="{self.size}">
        '''

class Range(Widget):
    def __init__(self, value=50, min=0, max=100):
        super().__init__()
        # databinding is defined using Property objects.
        self.value = Property(value)
        self.min = Property(min)
        self.max = Property(max)

    def widget_js(self):
        # Note that the 'input' event would enable during-drag feedback,
        # but this is pretty slow on google colab.
        return '''
          element.addEventListener('change', (e) => {
            model.set('value', element.value);
          });
          model.on('value', (value) => {
            if (!element.matches(':active')) {
              element.value = value;
            }
          })
        '''
    def widget_html(self):
        return f'''
          <input id="{self.view_id()}" type="range"
             value="{self.value}" min="{self.min}" max="{self.max}">
        '''

class Choice(Widget):
    def __init__(self, choices=None, horizontal=False):
        super().__init__()
        if choices is None:
            choices = []
        self.choices = Property(choices)
        self.horizontal = Property(horizontal)
        self.selection = Property(None)
    def widget_js(self):
        # Note that the 'input' event would enable during-drag feedback,
        # but this is pretty slow on google colab.
        return '''
          function esc(unsafe) {
            return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
          }
          function render() {
            console.log('rendering');
            var lines = model.get('choices').map((c) => {
              return '<label><input type="radio" name="choice" value="' +
                 esc(c) + '">' + esc(c) + '</label>'
            });
            element.innerHTML = lines.join(model.get('horizontal')?' ':'<br>');
          }
          model.on('choices horizontal', render);
          model.on('selection', (selection) => {
            [...element.querySelectorAll('input')].forEach((e) => {
              e.checked = (e.value == selection);
            })
          });
          element.addEventListener('change', (e) => {
            model.set('selection', element.choice.value);
          });
        '''
    def widget_html(self):
        radios = [
            f"""<label><input name="choice" type="radio"
            value="{html.escape(value)}">{html.escape(value)}</label>"""
            for value in self.choices ]
        sep = " " if self.horizontal else "<br>"
        return f'<form id="{self.view_id()}">{sep.join(radios)}</form>'

class Div(Widget):
    def __init__(self, innerHTML=''):
        super().__init__()
        # databinding is defined using Property objects.
        self.innerHTML = Property(innerHTML)

    def print(self, text, replace=False):
        newHTML = '<pre>%s</pre>' % html.escape(str(text));
        if replace:
            self.innerHTML = newHTML
        else:
            self.innerHTML += newHTML

    def widget_js(self):
        # Note that the 'input' event would enable during-drag feedback,
        # but this is pretty slow on google colab.
        return '''
          model.on('innerHTML', (innerHTML) => {
            element.innerHTML = innerHTML;
          })
        '''
    def widget_html(self):
        return f'''
          <div id="{self.view_id()}">{self.innerHTML}</div>
        '''

##########################################################################
## Implementation Details
##########################################################################

WIDGET_ENV = None
if WIDGET_ENV is None:
    try:
        from google.colab import output as colab_output
        WIDGET_ENV = 'colab'
    except:
        pass
if WIDGET_ENV is None:
    try:
        from ipykernel.comm import Comm as jupyter_comm
        get_ipython().kernel.comm_manager
        WIDGET_ENV = 'jupyter'
    except:
        pass

SEND_RECV_JS = """
function recvFromPython(obj_id, fn) {
  var recvname = "recv_" + obj_id;
  if (window[recvname] === undefined) {
    window[recvname] = new BroadcastChannel("channel_" + obj_id);
  }
  window[recvname].addEventListener("message", (ev) => {
    if (ev.data == 'ok') {
      window[recvname].ok = true;
      return;
    }
    fn.apply(null, ev.data.slice(1));
  });
}
function sendToPython(obj_id, ...args) {
  google.colab.kernel.invokeFunction('invoke_' + obj_id, args, {})
}
""" if WIDGET_ENV == 'colab' else """
function getChan(obj_id) {
  var cname = "comm_" + obj_id;
  if (!window[cname]) { window[cname] = []; }
  var chan = window[cname];
  if (!chan.comm && Jupyter.notebook.kernel) {
    chan.comm = Jupyter.notebook.kernel.comm_manager.new_comm(cname, {});
    chan.comm.on_msg((ev) => {
      if (chan.retry) { clearInterval(chan.retry); chan.retry = null; }
      if (ev.content.data == 'ok') { return; }
      var args = ev.content.data.slice(1);
      for (fn of chan) { fn.apply(null, args); }
    });
    chan.retry = setInterval(() => { chan.comm.open(); }, 2000);
  }
  return chan;
}
function recvFromPython(obj_id, fn) {
  getChan(obj_id).push(fn);
}
function sendToPython(obj_id, ...args) {
  var comm = getChan(obj_id).comm;
  if (comm) { comm.send(args); }
}
"""


WIDGET_MODEL_JS = SEND_RECV_JS + """
class Model {
  constructor(obj_id, init) {
    this._id = obj_id;
    this._listeners = {};
    this._data = Object.assign({}, init)
    recvFromPython(this._id, (name, value) => {
      this._data[name] = value;
      if (this._listeners.hasOwnProperty(name)) {
        this._listeners[name].forEach((fn) => { fn(value); });
      }
    })
  }
  trigger(name, value) {
    sendToPython(this._id, name, value);
  }
  get(name) {
    return this._data[name];
  }
  set(name, value) {
    this.trigger(name, value);
  }
  on(name, fn) {
    name.split(/\s+/).forEach((n) => {
      if (!this._listeners.hasOwnProperty(n)) {
        this._listeners[n] = [];
      }
      this._listeners[n].push(fn);
    });
  }
  off(name, fn) {
    name.split(/\s+/).forEach((n) => {
      if (!fn) {
        delete this._listeners[n];
      } else if (this._listeners.hasOwnProperty(n)) {
        this._listeners[n] = this._listeners[n].filter(
            (e) => { return e !== fn; });
      }
    });
  }
}
"""
