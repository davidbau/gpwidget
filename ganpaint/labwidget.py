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
    fn.apply(null, ev.data.slice(1));
  });
}
function sendToPython(obj_id, ...args) {
  google.colab.kernel.invokeFunction('invoke_' + obj_id, args, {})
}
""" if WIDGET_ENV == 'colab' else """
function getComm(obj_id) {
  var cname = "comm_" + obj_id;
  if (window[cname] === undefined) {
    var comm = Jupyter.notebook.kernel.comm_manager.new_comm(cname);
    comm.recvlist = [];
    comm.on_msg((ev) => {
      var args = ev.content.data.slice(1);
      for (fn of comm.recvlist) {
        fn.apply(null, args);
      }
    });
    window[cname] = comm;
  }
  return window[cname];
}
function recvFromPython(obj_id, fn) {
  getComm(obj_id).recvlist.push(fn);
}
function sendToPython(obj_id, ...args) {
  getComm(obj_id).send(args);
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
    // Don't set the model directly.  Python callback will do it.
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



class WidgetProperty(object):
    def __init__(self, value=None):
      self.value = value

class WidgetEvent(object):
    def __init__(self):
      return

class WidgetModel(object):
    def __init__(self):
        self._listeners = {}
        # In the jupyter case, we need to handle more details.
        # We could have multiple comms, and there can be some delay
        # between js injection and comm creation, so we need to queue
        # messages until the first comm is created.
        self._comms = []
        self._queue = []
        self._viewcount = 0
        def handle_remote_set(name, value):
            assert name in self._listeners, 'Protocol error: unknown ' + name
            setattr(self, name, value)
        self._recv_from_js(handle_remote_set)

    def on(self, name, cb):
        for n in name.split():
            if n not in self._listeners:
                raise ValueError(n + ' is not a WidgetProperty')
            self._listeners[n].append(cb)

    def off(self, name, cb):
        for n in name.split():
            if n not in self._listeners:
                raise ValueError(n + ' is not a WidgetProperty')
            self._listeners[n] = [c for c in self._listeners[n] if c != cb]

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

    def __setattr__(self, name, value):
        if isinstance(value, (WidgetProperty, WidgetEvent)):
            if self._listeners is None:
                raise ValueError("Must call super.__init__() first.")
            if name not in self._listeners:
                self._listeners[name] = []
        elif hasattr(self, name):
            curvalue = super().__getattribute__(name)
            if isinstance(curvalue, WidgetProperty):
                curvalue.value = value
                # Send to remote listeners
                self._send_to_js(id(self), name, value)
                # And local listeners also
                for cb in self._listeners[name]:
                    cb(value)
                return
            elif isinstance(curvalue, WidgetEvent):
                raise AttributeError("Cannot redefine an event.")
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        curvalue = super().__getattribute__(name)
        if isinstance(curvalue, WidgetProperty):
            return curvalue.value
        return curvalue

    def _send_to_js(self, *args):
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
            def open_comm(comm, open_msg):
                self._comms.append(comm)
                comm.on_msg(handle_comm)
                if self._queue:
                    for args in self._queue:
                        comm.send(args)
                    self._queue.clear()
                if open_msg['content']['data']:
                    handle_comm(open_msg)
            cname = "comm_" + str(id(self))
            get_ipython().kernel.comm_manager.register_target(cname, open_comm)
