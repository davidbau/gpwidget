from IPython.display import display, Javascript
from traitlets import Unicode, Bool, Int, Float, validate, TraitError
from ipywidgets import DOMWidget, register


@register
class PaintWidget(DOMWidget):
    def __init__(self, *args, **kwargs):
        inject_js_definition(PAINT_WIDGET_JS)
        super().__init__(*args, **kwargs)

    _view_name = Unicode('PaintView').tag(sync=True)
    _view_module = Unicode('paint_widget').tag(sync=True)
    _view_module_version = Unicode('0.0.1').tag(sync=True)
    
    mask = Unicode('', help="The mask as a data url.").tag(sync=True)
    image = Unicode('', help="The image as a url.").tag(sync=True)
    brushsize = Float(10.0, help='Brush radius').tag(sync=True)
    erase = Bool(False, help='Erase mode').tag(sync=True)
    disabled = Bool(False, help='Disabled interaction').tag(sync=True)
    width = Int(256, help="Canvas width.").tag(sync=True)
    height = Int(256, help="Canvas height.").tag(sync=True)



g_defined_widgets = {}
def inject_js_definition(js, redefine=False):
    # if not redefine and js in g_defined_widgets:
    #    return
    display(Javascript(js))
    g_defined_widgets[js] = True


PAINT_WIDGET_JS = '''
require.undef('paint_widget');
define('paint_widget', ["@jupyter-widgets/base"], function(widgets) {

    var PaintView = widgets.DOMWidgetView.extend({

        // Render the view.
        render: function() {
            this.size_changed();
            // Python -> JavaScript update
            this.model.on('change:mask', this.mask_changed, this);
            this.model.on('change:image', this.image_changed, this);
            this.model.on('change:width', this.size_changed, this);
            this.model.on('change:height', this.size_changed, this);
        },
        
        mouse_stroke: function() {
            var self = this;
            if (self.model.get('disabled')) { return; }
            function track_mouse(evt) {
                if (evt.type == 'keydown' || self.model.get('disabled')) {
                    console.log(evt);
                    if (self.model.get('disabled') || evt.which == 27 ||
                             evt.key === "Escape") {
                        $(window).off('mousemove mouseup keydown', track_mouse);
                        self.mask_changed();
                    }
                    return;
                }
                if (evt.type == 'mouseup' ||
                    (typeof evt.buttons != 'undefined' && evt.buttons == 0)) {
                    $(window).off('mousemove mouseup keydown', track_mouse);
                    self.model.set('mask', self.mask_canvas.toDataURL());
                    self.model.save_changes();
                    return;
                }
                var p = self.cursor_position();
                self.fill_circle(p.x, p.y,
                    self.model.get('brushsize'),
                    self.model.get('erase'));
            }
            this.mask_canvas.focus();
            $(window).on('mousemove mouseup keydown', track_mouse);
        },

        mask_changed: function(val) {
            this.draw_data_url(this.mask_canvas, this.model.get('mask'));
        },

        image_changed: function() {
            this.draw_data_url(this.image_canvas, this.model.get('image'));
        },
        
        size_changed: function() {
            this.mask_canvas = document.createElement('canvas');
            this.image_canvas = document.createElement('canvas');
            for (var attr of ['width', 'height']) {
                this.mask_canvas[attr] = this.model.get(attr);
                this.image_canvas[attr] = this.model.get(attr);
            }
            $(this.mask_canvas).css({position: 'absolute', top: 0, left:0,
                                    zIndex: '1'});
            this.el.innerHTML = '';
            this.el.appendChild(this.image_canvas);
            this.el.appendChild(this.mask_canvas);
            $(this.mask_canvas).on('mousedown', this.mouse_stroke.bind(this));
            this.mask_changed();
            this.image_changed();
        },

        cursor_position: function(evt) {
            const rect = this.mask_canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            return {x: x, y: y};
        },
        
        fill_circle: function(x, y, r, erase, blur) {
            var ctx = this.mask_canvas.getContext('2d');
            ctx.save();
            if (blur) {
                ctx.filter = 'blur(' + blur + 'px)';
            }
            ctx.globalCompositeOperation = (
                erase ? "destination-out" : 'source-over');
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fill();
            ctx.restore()
        },
        
        draw_data_url: function(canvas, durl) {
            var ctx = canvas.getContext('2d');
            var img = new Image;
            $(img).on('load error', function() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
            });
            img.src = durl;
        },
        
    });

    return {
        PaintView: PaintView
    };
});
'''
