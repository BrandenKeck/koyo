###
# TODO TODO TODO
###

import jinja2, pdfkit, shutil, json
import pandas as pd

class reports():

    def __init__(self):
        self.name = "report"
        self.report_data = None

    def set_name(self, name):
        self.name = name

    def generate(self, data):

        # Generate PDF Report
        context = {'name': self.name, 'games': data}
        options = {'enable-local-file-access': None}
        template_loader = jinja2.FileSystemLoader('./')
        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template('src/report_template.html')
        output_text = template.render(context)
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')
        pdfkit.from_string(output_text, f'F:/floating_repos/lakshmi/reports/{self.name}.pdf', configuration=config, options=options)
        shutil.copyfile(f'F:/floating_repos/lakshmi/reports/{self.name}.pdf', f'C:/Users/kril/Dropbox/Lakshmi/{self.name}.pdf')

        # Generate CSV Report
        bets = {"bet": [], "sb": [], "me": [], "kc": [], "type": []}
        for rd in data:
            odd = rd["odds"]  # Pull Data
            home = rd["home"]["name"]
            away = rd["away"]["name"]
            bets["bet"].append(f"{home} > {away}")  # Home ML
            bets["sb"].append(odd["Sports Book Home Win"])
            bets["me"].append(odd["Home Win Probability"])
            bets["kc"].append(odd["Home Win Kelly Criterion"])
            bets["type"].append("ML")
            bets["bet"].append(f"{away} > {home}")  # Away ML
            bets["sb"].append(odd["Sports Book Away Win"])
            bets["me"].append(odd["Away Win Probability"])
            bets["kc"].append(odd["Away Win Kelly Criterion"])
            bets["type"].append("ML")
            bets["bet"].append(f"{away} / {home} Over {odd['Sports Book Goals']}")  # Over
            bets["sb"].append(odd["Sports Book Over"])
            bets["me"].append(odd["Over Probability"])
            bets["kc"].append(odd["Over Kelly Criterion"])
            bets["type"].append("O")
            bets["bet"].append(f"{away} / {home} Under {odd['Sports Book Goals']}")  # Under
            bets["sb"].append(odd["Sports Book Under"])
            bets["me"].append(odd["Under Probability"])
            bets["kc"].append(odd["Under Kelly Criterion"])
            bets["type"].append("U")
        pd.DataFrame(bets).to_csv(f"./reports/{self.name}.csv")
        pd.DataFrame(bets).to_csv(f"C:/Users/kril/Dropbox/Lakshmi/{self.name}.csv")

        # Save Data to Last Report
        with open('last_report.json', 'w') as fp:
            json.dump({"report": data}, fp)

    # Regenerate Report from Saved Data
    def regenerate(self):
        self.set_name("Regenerated Report")
        self.load_last()
        self.generate(self.report_data)

    # Load Data from Last Report
    def load_last(self):
        f = open('last_report.json')
        self.report_data = json.load(f)["report"]
