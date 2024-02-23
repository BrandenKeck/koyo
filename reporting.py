import jinja2, pdfkit, shutil, json, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KoyoReport():

    def __init__(self):
        self.name = "report"

    def set_name(self, name):
        self.name = name

    def generate(self, data):

        # Generate Histograms
        pass
        
        # Generate PDF Report
        context = {'name': self.name, 'games': data}
        options = {'enable-local-file-access': None}
        template_loader = jinja2.FileSystemLoader('./')
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template('./report_template.html')
        output_text = template.render(context)
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        pdfkit.from_string(output_text, f'./report/{self.name}.pdf', configuration=config, options=options)
        # shutil.copyfile(f'F:/floating_repos/lakshmi/reports/{self.name}.pdf', f'C:/Users/kril/Dropbox/Lakshmi/{self.name}.pdf')

    def build_team_histogram(self, data):
        file = f'./reports/img/{str(uuid.uuid4())}.png'
        fig = plt.figure(figsize=(5, 5))
        plt.hist(data, bins=np.arange(13))
        plt.savefig(file)
        plt.close('all')
        return file