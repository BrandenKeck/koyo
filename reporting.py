import os, jinja2, pdfkit, json, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KoyoReport():

    def __init__(self):
        self.name = "report"

    def set_name(self, name):
        self.name = name

    def generate(self, data):

        # Empty Images Directory
        for f in os.listdir("./report/img/"):
            os.remove("./report/img/"+f)

        # Generate Histograms
        for idx, game in enumerate(data):
            data[idx]['home']['distribution'] = self.build_team_histogram(data[idx]["home_goals"])
            data[idx]['away']['distribution'] = self.build_team_histogram(data[idx]["away_goals"])
        
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
        file = f'./report/img/{str(uuid.uuid4())}.png'
        fig = plt.figure(figsize=(5, 5))
        plt.bar(np.arange(len(data)), data, tick_label=np.arange(len(data)))
        plt.yticks(color='w')
        plt.savefig(file)
        plt.close('all')
        return file