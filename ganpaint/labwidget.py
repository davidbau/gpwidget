"""
labwidget by David Bau.

Base class for a lightweight javascript notebook widget framework
that is portable across Google colab and Jupyter notebooks.
No use of requirejs: the design uses all inline javascript.

Defines WidgetModel and WidgetProperty, which set up data binding
using the communication channels available in either google colab
environment or jupyter notebook.  New can be defined as in the
following example.

class TextWidget(WidgetModel):
  def __init__(self, value='', width=20):
    super().__init__()
    # databinding is defined using WidgetProperty objects.
    self.value = WidgetProperty(value)
    self.width = WidgetProperty(width)
    self.esc = WidgetEvent()

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
       value="{html.escape(self.value)}" width="{self.width}">
    '''

User interaction should update the javascript model using
model.set('propname', value); this will propagate to the python
model and trigger any registered python listeners.

Communication pattern goes in a V shape:
User entry -> js model set -> python model ->  js model get -> User feedback

TODO: Support jupyterlab also.
"""

import json, html
from inspect import signature

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
class WidgetModel {
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
  get(name) {
    return this._data[name];
  }
  set(name, value) {
    // Do not set the model directly.  Python callback will do it.
    sendToPython(this._id, name, value);
  }
  trigger(name, value) {
    sendToPython(this._id, name, value);
  }
  on(name, fn) {
    name.split().forEach((n) => {
      if (!this._listeners.hasOwnProperty(n)) {
        this._listeners[n] = [];
      }
      this._listeners[n].push(fn);
    });
  }
  off(name, fn) {
    name.split().forEach((n) => {
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

class WidgetEvent(object):
    """
    Events provide a tree notification protocol where view interactions
    are requested at a leaf and the authoritative model sits at the root.
    Interaction requests are passed up to the root, and at the root, the
    handle method is called.  By default, the handle method just accepts
    the request and triggers handlers and notifications down the tree.
    """
    def __init__(self):
        self._listeners = []
        self.parent = None
    def handle(self, value):
        self.trigger(value)
    def request(self, value):
        if self.parent is not None:
            self.parent.request(value)
        else:
            self.handle(value)
    def set(self, value):
        if self.parent is not None:
            self.parent.off(self.handle)
            self.parent = None
        if isinstance(value, WidgetEvent):
            ancestor = value.parent
            while ancestor is not None:
                if ancestor == self:
                    raise ValueError('bound properties should not form a loop')
                ancestor = ancestor.parent
            self.parent = value
            self.parent.on(self.handle)
        elif not isinstance(self, WidgetProperty):
            raise ValueError('only properties can be set to a value')
    def trigger(self, value):
        for cb in self._listeners:
            if len(signature(cb).parameters) == 0:
                cb() # no-parameter callback.
            else:
                cb(value)
    def on(self, cb):
        self._listeners.append(cb)
    def off(self, cb):
        self._listeners = [c for c in self._listeners if c != cb]

class WidgetProperty(WidgetEvent):
    """
    A property is just a WidgetEvent that remembers its last value;
    setting a value makes it a root, and notifies children of the new value.
    """
    def __init__(self, value=None):
        super().__init__()
        self.set(value)
    def handle(self, value):
        self.value = value
        self.trigger(value)
    def set(self, value):
        super().set(value)
        if isinstance(value, WidgetProperty):
            value = value.value
        if not isinstance(value, WidgetEvent):
            self.handle(value)

class WidgetModel(object):
    def __init__(self):
        # In the jupyter case, we need to handle more details.
        # We could have multiple comms, and there can be some delay
        # between js injection and comm creation, so we need to queue
        # messages until the first comm is created.
        self._comms = []
        self._queue = []
        self._viewcount = 0
        def handle_remote_set(name, value):
            self.prop(name).request(value)
        self._recv_from_js(handle_remote_set)

    def on(self, name, cb):
        for n in name.split():
            self.prop(n).on(cb)

    def off(self, name, cb):
        for n in name.split():
            self.prop(n).off(cb)

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
                if isinstance(v, WidgetProperty)})
        return f"""
        {self.widget_html()}
        <script>
        (function() {{
        {WIDGET_MODEL_JS}
        var model = new WidgetModel("{id(self)}", {json_data});
        var element = document.getElementById("{self.view_id()}");
        {self.widget_js()}
        }})();
        </script>
        """

    def prop(self, name):
        curvalue = super().__getattribute__(name)
        if not isinstance(curvalue, WidgetEvent):
            raise AttributeError('%s not a property or event but %s' 
                    % (name, str(type(curvalue))))
        return curvalue

    def __setattr__(self, name, value):
        if hasattr(self, name):
            curvalue = super().__getattribute__(name)
            if isinstance(curvalue, WidgetEvent):
                # Delegte "set" to the underlying WidgetProperty.
                curvalue.set(value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
            if isinstance(value, WidgetEvent):
                if not hasattr(self, '_viewcount'):
                    raise ValueError('base WidgetModel __init__ must be called')
                def trigger_js(value):
                    self._send_to_js(id(self), name, value)
                if isinstance(value, WidgetEvent):
                    value.on(trigger_js)

    def __getattribute__(self, name):
        curvalue = super().__getattribute__(name)
        if isinstance(curvalue, WidgetProperty):
            return curvalue.value
        return curvalue

    def _send_to_js(self, *args):
        if self._viewcount > 0:
            if WIDGET_ENV == 'colab':
                google.colab.output.eval_js(f"""
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
            google.colab.output.register_callback(f"invoke_{id(self)}", fn)
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

##########################################################################
## Specific widgets
##########################################################################

class Button(WidgetModel):
    def __init__(self, label='button'):
        super().__init__()
        self.click = WidgetEvent()
        self.label = WidgetProperty(label)
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
            value="{html.escape(self.label)}">
        '''


class Textbox(WidgetModel):
  def __init__(self, value='', size=20):
    super().__init__()
    # databinding is defined using WidgetProperty objects.
    self.value = WidgetProperty(value)
    self.size = WidgetProperty(size)

  def widget_js(self):
    # Both "model" and "element" objects are defined within the scope
    # where the js is run.  "element" looks for the element with id
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
       value="{html.escape(self.value)}" size="{self.size}">
    '''
