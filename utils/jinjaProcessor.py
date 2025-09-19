from jinja2 import Environment, FileSystemLoader
import os

def process_template(template_file_path, context):
    template_dir = os.path.dirname(template_file_path)
    template_file = os.path.basename(template_file_path)
    env = Environment(loader=FileSystemLoader(template_dir),auto_reload=True)
    template = env.get_template(template_file)
    return template.render(**context)

def process_template_no_var(template_file_path):
    template_dir = os.path.dirname(template_file_path)
    template_file = os.path.basename(template_file_path)
    env = Environment(loader=FileSystemLoader(template_dir),auto_reload=True)
    template = env.get_template(template_file)
    return template.render()