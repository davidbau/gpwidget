"""
colabwidget

Defines WidgetModel and WidgetProperty, which set up data binding
using the communication channels available in the google colab
environment.  New can be defined as in the following example.

class TextWidget(WidgetModel):
  def __init__(self):
    super().__init__()
    # After super().init(), databinding is done using WidgetProperty objects.
    self.value = WidgetProperty('')
    self.width = WidgetProperty(20)

  def widget_js(self):
    # A special javascript "model" object is provided
    return '''
      var elt = document.getElementById("%s");
      elt.value = model.get('value');
      elt.width = model.get('width');
      elt.addEventListener('keydown', (e) => {
        if (e.code == 'Enter') {
          model.set('value', elt.value);
        }
      });
      model.on('value', (value) => {
        elt.value = model.get('value');
      });
      model.on('width', (value) => {
        elt.width = model.get('width');
      });
    ''' % self.view_id()
  def widget_html(self):
    return f'''
    <input id="{self.view_id()}"
       value="{html.escape(self.value)}" width="{self.width}">
    '''

User interaction should update the javascript model using
model.set('propname', value); this will propagate to the python
model and trigger any registered python listeners.


TODO: 
"""

import json, html
from google.colab import output

WIDGET_MODEL_JS = """
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
    if (!this._listeners.hasOwnProperty(name)) {
      this._listeners[name] = [];
    }
    this._listeners[name].push(fn);
  }
  off(name, fn) {
    if (!fn) {
      delete this._listeners[name];
    } else if (this._listeners.hasOwnProperty(name)) {
      this._listeners = this._listeners.filter((e) => { return e !== fn; });
    }
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
    self._viewcount = 0
    def handle_remote_set(name, value):
      assert name in self._listeners, 'Protocol error: notified on ' + name
      setattr(self, name, value)
      for cb in self._listeners[name]:
        cb(value)
    self._recv_from_js(handle_remote_set)

  def on(name, cb):
    assert name in self._listeners, 'Can only listen to WidgetProperties'
    self._listeners[name].append(cb)

  def widget_html(self):
    return ''
  
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
    {self.widget_js()}
    }})();
    </script>
    """

  def __setattr__(self, name, value):
    if isinstance(value, (WidgetProperty, WidgetEvent)):
      assert self._listeners is not None, "Must call super.__init__() first."
      if name not in self._listeners:
        self._listeners[name] = []
    elif hasattr(self, name):
      curvalue = super().__getattribute__(name)
      if isinstance(curvalue, WidgetProperty):
        curvalue.value = value
        self._send_to_js(id(self), name, value)
        return
      elif isinstance(curvalue, WidgetEvent):
        assert False, "Cannot redefine an event."
    super().__setattr__(name, value)

  def __getattribute__(self, name):
    curvalue = super().__getattribute__(name)
    if isinstance(curvalue, WidgetProperty):
      return curvalue.value
    return curvalue
  
  def _send_to_js(self, *args):
    output.eval_js(f"""
    (window.send_{id(self)} = window.send_{id(self)} ||
    new BroadcastChannel("channel_{id(self)}")).postMessage({json.dumps(args)});
    """, ignore_result=True)

  def _recv_from_js(self, fn):
    output.register_callback(f"invoke_{id(self)}", fn)
