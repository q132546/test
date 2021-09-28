import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('pre_total.csv')
print(data.columns)


class DataProcess():
    def __init__(self):
        self.pre_total_pd = pd.read_csv('pre_total.csv')
        self.post_education_total_pd = pd.read_csv('post_education_total.csv')

    def merge_data(self):
        # post apn & rn merge
        post_education_apn_pd = pd.read_csv('post_education_APN.csv')
        post_education_rn_pd = pd.read_csv('post_education_RN.csv')

        post_education_total_pd = pd.concat([post_education_rn_pd, post_education_apn_pd], axis=0)

        # save post total data
        post_education_total_pd.to_csv('post_education_total.csv', index=False)

    @staticmethod
    def education_answer(number):
        if number == 1:
            return {'question': '1. According to American Academy of Pediatrics (AAP), what is the ideal sound level in NICU?', 'ans': 'Below 45 dB'}

        elif number == 2:
            return {'question': '2. When does a fetus start to hear sounds?', 'ans': '17 weeks of gestational age'}

    @staticmethod
    def education_item(number):
        if number == 1:
            return {'Below 40 dB': 1, 'Below 45 dB': 2, 'Below 50 dB': 3, 'Below 55 dB': 4}

        elif number == 2:
            return {'12 weeks of gestational age': 1, '17 weeks of gestational age': 2, '22 weeks of gestational age': 3, '27 weeks of gestational age': 4}

    def main_process(self):
        for num in [2]:
            question = self.education_answer(num)['question']
            answer = self.education_answer(num)['ans']

            pre_temp_education_np = np.array(self.pre_total_pd[question])
            post_temp_education_np = np.array(self.post_education_total_pd[question])

            for i in range(len(pre_temp_education_np)):
                pre_temp_education = pre_temp_education_np[i]
                post_temp_education = post_temp_education_np[i]

                plt.subplot(2, 1, 1)
                plt.scatter(i, self.education_item(num)[pre_temp_education], c='r', marker='o')

                plt.subplot(2, 1, 2)
                plt.scatter(i, self.education_item(num)[pre_temp_education], c='r', marker='o')

                if self.education_item(num)[pre_temp_education] != self.education_item(num)[post_temp_education]:
                    plt.scatter(i, self.education_item(num)[post_temp_education], c='b', marker='o')

            plt.show()


if __name__ == '__main__':
    data_process = DataProcess()
    data_process.main_process()
