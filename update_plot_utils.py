import re

with open('whatsapp_analyzer/plot_utils.py', 'r') as f:
    content = f.read()

helpers = """
import json
import uuid

def wrap_base64_img(img_base64):
    return f'<img src="data:image/png;base64,{img_base64}" alt="Plot" style="max-width: 100%; height: auto; border-radius: 8px;">'

def render_chartjs(config):
    chart_id = "chart_" + uuid.uuid4().hex[:8]
    html = f'''
    <div style="position: relative; height: 300px; width: 100%;">
        <canvas id="{chart_id}"></canvas>
    </div>
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var ctx = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx, {json.dumps(config)});
    }});
    </script>
    '''
    return html
"""

content = content.replace('import base64\n', 'import base64\n' + helpers)

content = content.replace('def plot_to_base64(plt):', 'def plot_to_base64(plt, wrap=True):\n    """Convert a Matplotlib plot to a base64 encoded image."""\n    img_buffer = BytesIO()\n    plt.savefig(img_buffer, format="png", bbox_inches="tight")\n    img_buffer.seek(0)\n    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")\n    plt.close()\n    return wrap_base64_img(img_base64) if wrap else img_base64\n\n# Old plot_to_base64 removed')

# Use regex to remove the old plot_to_base64 implementation properly since the replace above might leave the old body
content = re.sub(r'# Old plot_to_base64 removed.*?(?=def _render_empty_plot)', '', content, flags=re.DOTALL)

with open('whatsapp_analyzer/plot_utils.py', 'w') as f:
    f.write(content)
