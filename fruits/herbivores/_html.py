import numpy as np

def to_arr_header(data):
    orig_keys = []
    for key in data:
        if hasattr(data[key], "shape") and len(data[key].shape) == 1:
            orig_keys.append(key)
    keys = ['/'.join(k) for k in orig_keys]
    shape = (len(data[orig_keys[0]]), len(orig_keys))
    for k in orig_keys:
        assert data[k].shape == (shape[0],)
    result = np.empty(shape, dtype=np.object)
    for i, k in enumerate(orig_keys):
        result[:,i] = data[k].astype(np.str)
    return result, np.array(keys)

def columns_width(arr, header, maximum):
    width = []
    for i, key in enumerate(header):
        l = [len(s) for s in arr[:,i]] + [len(key)]
        width.append(max(np.max(l)/3, np.mean(l)))
    width = np.array(width, dtype=np.float)
    width = np.minimum(width, maximum)
    width /= np.sum(width)
    return np.ceil(100 * width)

def to_html(arr, header, width, class_name):
    hs = ["<table class={}><thead><tr>".format(class_name)]
    for w, key in zip(width, header):
        hs.append("<th width={}%>{}</th>".format(w, key))
    hs.append("</tr></thead><tbody>")
    for i in range(arr.shape[0]):
        hs.append("<tr>")
        for j in range(arr.shape[1]):
            hs.append("<td>{}</td>".format(arr[i,j]))
        hs.append("</tr>")
    hs.append("</tbody></table>")
    return '\n'.join(hs)

def sanitize_html(text):
    import re
    def _sanitize(s):
        return re.sub(">", "&gt;", re.sub("<", "&lt;", re.sub("&", "&amp;", s)))
    if isinstance(text, str):
        return _sanitize(text)
    elif isinstance(text, np.ndarray):
        return np.vectorize(_sanitize, otypes=(np.object,))(text)
    else:
        raise TypeError(text)

doc_template = lambda head, body: "<!DOCTYPE html><html><head>{}</head><body>{}</body></html>".format(head, body)

style_template = lambda *styles: "<style>{}</style>".format('\n'.join(styles))

div_template = lambda text, class_name=None: "<div{class_name}>{text}</div>".format(text=text, class_name=(' class="'+class_name+'"' if class_name is not None else ""))

table_style_1 = lambda class_name=None: """
table{class_name} {{ width: 100%; border: 0; border-collapse: collapse; table-layout: fixed; }}
table{class_name} th {{
    border: 1px solid #ddd;
    padding-top: 5px;
    padding-bottom: 5px;
    background-color: #4CAF50;
    color: white;
    text-align: center;
}}
table{class_name} td {{
    border: 1px solid #ddd;
    padding: 3px;
}}
table{class_name} tr:nth-child(even){{background-color: #f2f2f2;}}
table{class_name} tr:hover {{background-color: #ddd;}}
table{class_name}, table{class_name} thead, table{class_name} tbody, table{class_name} tr, table{class_name} th, table{class_name} td {{ word-wrap:break-word }}
""".format(class_name=('.'+class_name if class_name is not None else ""))

table_style_2 = lambda class_name=None: """
table{class_name} {{ border: 0; border-collapse: collapse; table-layout: fixed; }}
table{class_name} td {{
    border: 1px solid #ddd;
    padding: 3px;
}}
""".format(class_name=('.'+class_name if class_name is not None else ""))


div_style_1 = lambda class_name=None: """
div{} {{
    padding-left: 5px;
    padding-top: 15px;
    padding-bottom: 10px;
}}
""".format('.'+class_name if class_name is not None else "")

href = lambda address, text: "<a href=\"{}\">{}</a>".format(address, text)

tablesorter = lambda: """
<script src="http://code.jquery.com/jquery-latest.min.js"></script>
<script src="https://cdn.rawgit.com/christianbach/tablesorter/master/jquery.tablesorter.min.js"></script>
<!-- see https://github.com/christianbach/tablesorter for license -->
<script>
$(document).ready(function() 
    { 
        $("table").tablesorter(); 
    } 
);
</script>
"""
