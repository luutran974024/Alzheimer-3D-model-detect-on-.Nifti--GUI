import PIL.Image
import torch.nn
import random
from io import BytesIO
from hot_img import hot_img

import pywebio


from model import *
import nibabel as nib
import data.datasets
import time

from pyecharts.charts import Bar, Page
from pyecharts.globals import ThemeType
from pyecharts import options as opts

def bar_base(data) -> Bar:
    c = (
        Bar()
        # Bar({"theme": ThemeType.MACARONS})
        .add_xaxis(["CN", "MCI", "AD", "VMCI"])
        .add_yaxis("output_value", data, markpoint_opts=["max"])
        .set_global_opts(
            title_opts={"text": "Model output", "subtext": ""},)
        )
    return c


def refresh():
    pywebio.output.clear()
    page1()


def generate_random_str(target_length=32):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(target_length):
        random_str += base_str[random.randint(0, length)]
    return random_str


def make_image_list(path, user_ip, hot_type="LRP"):
    img_list = []
    x = 10
    for i in range(1, x):
        path_group = hot_img(path, 64 + 16 * i, hot_type)
        img_list.append(list(path_group))
    print_logs("make_img," + str(img_list[1:][0]).strip("[").strip("]")+",\n", user_ip)
    return img_list


def show_img_s(path, user_ip, mod, hot_type="LRP"):
    pywebio.output.popup("Image rendering can take a long time，Please wait", [pywebio.output.put_row(
        [pywebio.output.put_loading(shape="grow", color="success")],
    )])
    path_group = list(hot_img(path, 64 + 16 * 7, hot_type))
    print_logs("make_img," + str(path_group[0]).strip("[").strip("]") + ",\n", user_ip)
    img_table = None
    if mod == 3:
        for j in range(3):
            path_group[j] = PIL.Image.open(path_group[j])
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(path_group[2])],
        ])
    if mod == 1:
        for j in range(3):
            path_group[j] = PIL.Image.open(path_group[j])
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(path_group[0])],
        ])
    if mod == 2:
        for j in range(3):
            path_group[j] = PIL.Image.open(path_group[j])
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(path_group[1])],
        ])
    return pywebio.output.popup(title='image', content=img_table)


def show_img(path, user_ip, mod):
    pywebio.output.popup("Image rendering may take a long time, please be patient", [pywebio.output.put_row(
        [pywebio.output.put_loading(shape="grow", color="success")],
    )])
    img_list = make_image_list(path, user_ip)
    for i in img_list:
        for j in range(3):
            i[j] = PIL.Image.open(i[j])

    img_table = []
    if mod == 1:
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(img_list[8][0]), pywebio.output.put_image(img_list[7][0]),
             pywebio.output.put_image(img_list[6][0])],
            [pywebio.output.put_image(img_list[5][0]), pywebio.output.put_image(img_list[4][0]),
             pywebio.output.put_image(img_list[3][0])],
            [pywebio.output.put_image(img_list[2][0]), pywebio.output.put_image(img_list[1][0]),
             pywebio.output.put_image(img_list[0][0])],
        ])

    if mod == 2:
        img_table = pywebio.output.put_table([

            [pywebio.output.put_image(img_list[8][1]), pywebio.output.put_image(img_list[7][1]),
             pywebio.output.put_image(img_list[6][1])],
            [pywebio.output.put_image(img_list[5][1]), pywebio.output.put_image(img_list[4][1]),
             pywebio.output.put_image(img_list[3][1])],
            [pywebio.output.put_image(img_list[2][1]), pywebio.output.put_image(img_list[1][1]),
             pywebio.output.put_image(img_list[0][1])],

        ])
    if mod == 3:
        img_table = pywebio.output.put_table([
            [pywebio.output.put_image(img_list[8][2]), pywebio.output.put_image(img_list[7][2]),
             pywebio.output.put_image(img_list[6][2])],
            [pywebio.output.put_image(img_list[5][2]), pywebio.output.put_image(img_list[4][2]),
             pywebio.output.put_image(img_list[3][2])],
            [pywebio.output.put_image(img_list[2][2]), pywebio.output.put_image(img_list[1][2]),
             pywebio.output.put_image(img_list[0][2])]
        ])
    return pywebio.output.popup(title='image', content=img_table)


def compare_ans(a, ans):
    if a == ans:
        return True
    else:
        return False


def print_logs(content, user_ip):
    with open("./run_logs/" + user_ip+"run_logs.csv", 'a') as file:
        file.write(str(pywebio.session.info.user_ip) + "," +
                   str(pywebio.session.info.user_agent.device.model) + "," +
                   str(pywebio.session.info.user_agent.browser.family) + "," + time.ctime() + ",")
        file.write(content)


@pywebio.config(title="Demo", description="Alzheimer's disease diagnosis based on ADNI dataset",)
def page1(is_demo=False):
    user_ip = str(pywebio.session.info.user_ip)+generate_random_str(16)
    ans = "cognitive normal(CN)"
    ans_list = ["Alzheimer's disease(AD)", "cognitive normal(CN)", "mild cognitive impairment(MCI)", "very mild cognitive impairment(VMCI)"]

    ans_y = [1, 0, 0, 0]

    chart_html = bar_base(ans_y).render_notebook()

    temp_file_path = "demo.nii"

    graph_img = PIL.Image.open("./data/net_graph.png")
    # front_img = PIL.Image.open("./data/front_page1.png")
    train_img = PIL.Image.open("./data/train_process2.png")
    brain_img = PIL.Image.open("./data/brain_demo.png")

    hot_img9 = PIL.Image.open("./data/hot_img9.3.png")
    hot_img1 = PIL.Image.open("./data/hot_img1.3.png")
    # hot_only = PIL.Image.open("./data/hot_only.png")
    brain_demo1 = PIL.Image.open("./data/brain_demo1.png")

    while 1:
        try:
            pywebio.output.put_warning("The identification results are for reference only", closable=True, position=- 1)

            pywebio.output.put_html("<h1><center>Alzheimer's disease diagnosis based on ADNI dataset</center></h1><hr>")

            cn_content = [pywebio.output.put_markdown("Cognitive normal")]
            vmci_content = [pywebio.output.put_markdown("Very mild cognitive impairment")]
            mci_content = [pywebio.output.put_markdown("Mild cognitive impairment")]
            ad_content = [pywebio.output.put_markdown("Alzheimer's disease")]

            pywebio.output.put_row(
                [pywebio.output.put_scope(name="chart", content=[pywebio.output.put_html(chart_html)])
                 ],
            )

            pywebio.output.put_row(
                [pywebio.output.put_collapse("CN", cn_content, open=compare_ans("Cognitive normal(CN)", ans)),
                 pywebio.output.put_collapse("VMCI", vmci_content, open=compare_ans("Early mild cognitive impairment(VMCI)", ans)),
                 pywebio.output.put_collapse("MCI", mci_content, open=compare_ans("Mild cognitive impairment(MCI)", ans)),
                 pywebio.output.put_collapse("AD", ad_content, open=compare_ans("Alzheimer's disease(AD)", ans))],
            )

            more_content = [
                pywebio.output.put_table([
                    [
                        pywebio.output.put_image(hot_img9),
                        pywebio.output.put_image(hot_img1),
                    ],
                    [
                        pywebio.output.put_image(brain_img),
                        pywebio.output.put_image(brain_demo1),
                    ],
                ])
            ]
            f = open("model.py", "r", encoding="UTF-8")
            code = f.read()
            f.close()
            pywebio.output.put_collapse("Heat map demo", more_content, open=True, position=- 1)
            pywebio.output.put_row([
                pywebio.output.put_collapse("Model information", [pywebio.output.put_image(graph_img)], open=True, position=- 1),
                pywebio.output.put_collapse("training information", [pywebio.output.put_image(train_img),
                                                         pywebio.output.put_markdown(
                                                             "learning_rate=1e-4 weight_decay=1e-5"),
                                                         pywebio.output.put_markdown("batch_size=4 num_works=1"), ],

                                            open=True, position=- 1)
            ])
            pywebio.output.put_collapse("model code", [pywebio.output.put_code(code, "python")], open=False, position=- 1)
            pywebio.output.put_markdown("ref: https://github.com/moboehle/Pytorch-LRP")
            pywebio.output.put_markdown("datasets: https://adni.loni.usc.edu")

            action = pywebio.input.actions(' ',
                                           [{'label': "Upload .jpg image", 'value': "Upload .jpg image", 'color': 'warning'},
                                            {'label': "Use demo.nii", 'value': "Use demo.nii", 'color': 'info'},
                                            "View image"
                                            ])
            if action == "Use demo.nii":
                is_demo = True
            if action == "Upload .jpg image":
                is_demo = False
            if action == "View image":
                action = pywebio.input.actions(' ',
                                               ["View original photo",
                                                "View heat map",
                                                "View single layer original image",
                                                "View single layer heat map",
                                                {'label': "custom view", 'value': "custom view", 'color': 'dark',
                                                 "disabled": True}
                                                ])
                if action == "View original photo":
                  
                    show_img(temp_file_path, user_ip, 1)
                    pywebio.output.clear()
                    continue
                if action == "View heat map":
                 
                    show_img(temp_file_path, user_ip, 3)
                    pywebio.output.clear()
                    continue
                if action == "View single layer original image":
                    show_img_s(temp_file_path, user_ip, mod=1)
                    pywebio.output.clear()
                    continue
                if action == "View single layer heat map":
                    show_img_s(temp_file_path, user_ip, mod=3)
                    pywebio.output.clear()
                    continue

            ###################################################################################

            if is_demo is False:
                try:
                    inpic = pywebio.input.file_upload(label="Upload medical images(.jpg)")
                    inpic = BytesIO(inpic['content'])
                    temp_file_path = "./jpg/" + generate_random_str() + ".jpg"
                    with open(temp_file_path, 'wb') as file:
                        file.write(inpic.getvalue())  
                    print_logs("upload_file," + temp_file_path + ",\n", user_ip)
                except:
                    pywebio.output.toast("Input error, please upload medical imaging files(.jpg)", color="warn")
                    refresh()

            if is_demo is True:
                is_demo = False
                temp_file_path = "demo.nii"

            pywebio.output.popup("AI recognition in progress", [pywebio.output.put_row(
                [pywebio.output.put_loading(shape="grow", color="success")],
            )])

            ##############################################################################

            torch.no_grad()
            test_model = torch.load("./data/model_save/myModel_130.pth", map_location=torch.device('cpu'))
            test_model.eval()
            # print(test_model)

            img = None
            try:
                img = nib.load(temp_file_path)
                img = img.get_fdata()
                img = data.datasets.process_img(img)
                img = img.reshape((1, 1, -1, 256, 256))
                # print(img.shape)
            except Exception:
                pywebio.output.toast("Input processing error, please upload medical imaging files(.jpg)\tshould be greater than：(168x168)", color="warn")
                refresh()

            try:
                output = None
                with torch.no_grad():
                    output = test_model(img)
                ans_y = output.squeeze().tolist()
            except Exception:
                pywebio.output.toast("Model recognition error, possibly due to insufficient server memory, please try again later.", color="warn")
                refresh()

            # print(output)
            if min(ans_y) < 0:
                m = min(ans_y)
                for i in range(len(ans_y)):
                    ans_y[i] -= 1.2 * m
            ans = ans_list[output.argmax(1).item()]
            # print(ans)

            ######################################################################

            chart_html = bar_base([ans_y[1], ans_y[3], ans_y[2], ans_y[4], ans_y[0]]).render_notebook()
            with pywebio.output.use_scope(name="chart") as scope_name:
                pywebio.output.clear()
                pywebio.output.put_html(chart_html)
            # print(chart_html)

            show_result = [pywebio.output.put_markdown("diagnosed as:\n # " + ans)]
            pywebio.output.popup(title='AI recognition results', content=show_result)

            pywebio.output.clear()

        except Exception:
            continue

def bar_base(data) -> Bar:
    c = (
        Bar()
        .add_xaxis(["CN", "VMCI", "MCI", "AD"])
        .add_yaxis("output_value", data, itemstyle_opts=opts.ItemStyleOpts(color="#8A2BE2"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Model output", subtitle=""),
        )
    )
    return c

if __name__ == "__main__":
    # page1()
    pywebio.platform.start_server(
        applications=[page1, ],
        debug=False,
        auto_open_webbrowser=False,
        remote_access=False,
        cdn=False,
        port=27017
    )

