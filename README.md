# SMU-Project-1
Project 1 for SMU-Data Science
Caffeine Project Write-Up
Alexandra Blitch
Fabi Estrada
Cagan Abney


Our group selected the “Caffeine Content of Drinks” dataset from Kaggle for this project. Our inspiration for this dataset was based on our familiarity with caffeinated drinks. We also believe it is important to know the caffeine and calorie amounts in today’s popular drinks. This dataset is 29.21 kB which consists of 5 columns and 611 rows. The data includes volume of drinks (mL), calorie content, caffeine content (mg), drink name, and type.


After selecting our dataset, we began thinking of research questions. Our groups decided on the following research questions:

Does coffee/tea have a lower volume-to-caffeine ratio than other drink categories?
Does soda have a higher calorie-to-caffeine ratio than other drink categories?
Which drink type has the most caffeine per 100 mL?

After listing our research questions, we began writing our code. We started by importing modules into the notebook. The modules imported are as follows: 

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import scipy.stats as st

Below the modules, we added the colors for our visualizations and assigned them as variables as we wanted to use specific colors and easily call on them. In the next cell, we added and read the CSV from our dataset. We then ran a df.head() to confirm this worked. We then wanted to change the column names since some were lowercase and some were not. We changed “Volume (ml)” to “volume” and  “Caffeine (mg)” to “caffeine”.  Once the columns were renamed, we then ran a df.info() to check for null values. Our dataset came back with zero null values.

In the next cell, we ran a df.describe() to get a grasp on the averages, min, max, and quartiles of our data. The results are shown below:





volume

calories

caffeine
count
610.000000
610.00000
610.000000
mean
346.543630
75.527869
134.693443
std
143.747738
94.799919
155.362861
min
7.393375
0.000000
0.000000
25%
236.588000
0.000000
50.000000
50%
354.882000
25.000000
100.000000
75%
473.176000
140.00000
160.000000
max
1419.52800
830.00000
1555.00000



After viewing the table above, we noticed that the max volume, max calories, and max caffeine were oddly high. In the next three cells in our notebook, we sorted by values of each category to show us these outliers. This strategy was very insightful as it showed us that there may be some drinks that should not be in this dataset, for example, the ‘Starbucks Bottled Iced Coffee’, which has a volume of over 1419 mL. This is likely because it is a package or bundle of drinks rather than one single drink. We also saw that the drink with max calories was the ‘Arby’s Jamocha Shake’, which had a relatively low level of caffeine and normal volume, so we decided to keep it in our dataset as it is representative of coffee-type drinks on the market.

The last step of our initial data engineering was to check for duplicate values. We found that there were no duplicate values in our dataset and we were ready to complete our first visualization. Below you can see a bar graph of the total drinks by type: 

![bargraph](https://github.com/Cagan124/SMU-Project-1/assets/145722674/205f38b1-0fbc-40ca-b4ee-8068cf86d292)




After creating this bar graph, we decided that the data would be better displayed by combining the three smaller drink types with the three larger drink types. In our notebook, we used the df.loc() to combine Coffee and Tea, Energy Drinks and Energy Shots, and Water and Soda. Once we combined these categories we realized that Water and Soda ran bimodally and would need to be their own parent categories. After settling for these four parent categories we wanted to show these categories by percentages. The new drink categories can be seen below: 

![donutgraph](https://github.com/Cagan124/SMU-Project-1/assets/145722674/923f40dc-3036-4bd7-bef4-3554e42fd010)


We then created some bar graphs comparing the average amount of Caffeine and Calories to Caffeine and Calories per 100 mL. Coffee and Tea’s average caffeine dropped significantly when we compared it to the average Caffeine per 100mL. Soda had significantly higher Calories in both the average and the 100mL. The bar graphs are shown below:
![avgcaffeine](https://github.com/Cagan124/SMU-Project-1/assets/145722674/79826949-59c9-41ca-8300-a810986804af)
![caffeine100ml](https://github.com/Cagan124/SMU-Project-1/assets/145722674/0d6579b9-0c3d-4a4e-aadb-264fdf967b1d)
![avgcalories](https://github.com/Cagan124/SMU-Project-1/assets/145722674/6d5dcd7b-c9b6-4e35-8bb8-519017a95c12)
![calories100ml](https://github.com/Cagan124/SMU-Project-1/assets/145722674/4a5cceb7-0eda-496a-bd50-8465d4a61072)


	
	This led us to more questions: how strong is the correlation between calories and volume compared to our original dataset? What about caffeine and volume? Lastly, calories and caffeine? 
Below is the original dataset:
 ![table 1](https://github.com/Cagan124/SMU-Project-1/assets/145722674/b91cd15b-d650-4295-bee5-40443f028a0a)

Below, we can compare the original dataset to Coffee and Tea, Energy Drinks and Shots, Soda, and Water:
![coffee:tea](https://github.com/Cagan124/SMU-Project-1/assets/145722674/d734724b-d0dd-4411-b2e4-dfb645f5c580)

 
According to the chart, calories and volume had a stronger correlation than the original dataset, as did caffeine and volume. However, calories and caffeine had a slightly weaker correlation.

![energy drink](https://github.com/Cagan124/SMU-Project-1/assets/145722674/8dde4c61-2a2e-45f2-a3d3-ff8172827be4)



Energy Drinks and Shots seemed to have a stronger correlation between calories and caffeine, caffeine and volume seemed to be pretty even, and calories and volume had a slightly weaker correlation.
![soda](https://github.com/Cagan124/SMU-Project-1/assets/145722674/571048dd-5e41-4cdf-a88b-c9ac4e240c52)

Soda’s chart appeared to have a weaker correlation in both calories and caffeine when compared to volume, though calories and caffeine seemed to show a slightly stronger correlation when compared to each other.
![water](https://github.com/Cagan124/SMU-Project-1/assets/145722674/fab798c6-f311-4d67-90f5-d3f09aa8ebc1)

Both calories and caffeine compared to volume had a much weaker correlation than the original dataset, though, like Soda, calories and caffeine had a slightly stronger correlation.
After completing the correlations, we made a scatter plot to show Calories vs. Caffeine as shown below:
![calories v caffeine](https://github.com/Cagan124/SMU-Project-1/assets/145722674/6505a0c1-38a8-496d-9f6e-077f16d2d441)

This graph shows us that Coffee/Tea was lower in calories and higher in caffeine on average, Energy Drinks and Shots were about equal in calories and higher in caffeine, Soda was higher in calories and lower in caffeine, and Water was lower in calories and caffeine on average.

	Next, we made violin graphs that compare calories, caffeine, and volume. As stated above, Water and Soda ran bimodally; you can see below that the datasets were just too different to combine. Also, Coffee and Tea seemed to be way higher across the board, and Soda has more calories in comparison to volume and caffeine.


![caloriesviolin](https://github.com/Cagan124/SMU-Project-1/assets/145722674/335e8d78-4cfd-44b3-ad1d-b150e3abb31f)

![caffeineviolin](https://github.com/Cagan124/SMU-Project-1/assets/145722674/30f7d2fe-a2df-4784-8dbc-0803d4a99619)

![volumeviolin](https://github.com/Cagan124/SMU-Project-1/assets/145722674/2d46fbb7-1b79-4ed1-bf71-95d7a0e04b66)


We also did some T-Tests to see how different the caffeine levels were statistically. You can see in the first T-Test, that Coffee and Tea, and Energy Drinks and Shots were pretty similar, though Energy Drinks and Shots were significantly different than Water. Soda and Water also seem to be statistically insignificant.
![1st ttest](https://github.com/Cagan124/SMU-Project-1/assets/145722674/7a38e353-1779-4e94-9221-6758e533af18)

![2nd ttest](https://github.com/Cagan124/SMU-Project-1/assets/145722674/caebde01-b3ed-4473-a7bb-9ce48c17ac91)

![3rd ttest](https://github.com/Cagan124/SMU-Project-1/assets/145722674/570f4e3e-23db-43c8-9df4-54eec102ad18)



	Next, we looked back at our research questions to see what answers we could gather. For our first research question, “Does coffee/tea have a lower volume-to-caffeine ratio than other drink categories?”, we discovered that as expected coffee/tea has a lower volume-to-caffeine ratio than other categories. This is because many of the coffee/tea drinks were extremely high in caffeine.
![main regression](https://github.com/Cagan124/SMU-Project-1/assets/145722674/70511719-c41b-43cd-84f5-41dc3da1d0ab)


Similarly in the second research question, “Does soda have a higher calorie-to-caffeine ratio than other drink categories?”, our results showed the expected, that yes soda does have a higher calorie-to-caffeine ratio than other drink categories. Soda’s caffeine levels remained low while some of the sodas were very high in calories. 
![calories v caffeine](https://github.com/Cagan124/SMU-Project-1/assets/145722674/e2fededb-4c98-4445-9173-fa0cf4ab3e28)

	For the third research question, “Which drink type has the most caffeine per 100 mL?”, we discovered that energy drinks/shots had the most caffeine per 100 mL. This is likely caused by the fact that most of the energy shots are less than 100 mL, so comparing 100 mL quantities shows even higher caffeine levels than would be in a typical serving of an energy shot.
![caffeine100ml](https://github.com/Cagan124/SMU-Project-1/assets/145722674/692fb2a6-e2f2-472d-9ea0-991dc6a4b5e4)


To further evaluate our dataset, we ran a regression comparing caffeine and volume. Our regression showed that the correlation between caffeine and volume is not very statistically significant. All drink types except coffee/tea had a weak correlation, so this result is expected. For example, energy drinks/shots all had relatively similar levels of caffeine despite the energy shots having very low volume levels and the energy drinks having higher volume levels. 
![main regression](https://github.com/Cagan124/SMU-Project-1/assets/145722674/41fc0d51-e0cf-4d5e-9f6f-910c8f72b87e)

To confirm this, we ran regressions for each of the categories individually:
![coffee regression](https://github.com/Cagan124/SMU-Project-1/assets/145722674/66012b8a-a206-4621-b879-47bd1c6f3e85)
![energy regression](https://github.com/Cagan124/SMU-Project-1/assets/145722674/8022c976-cd04-464e-aba5-b1e0ed2fb4d1)
![soda regression](https://github.com/Cagan124/SMU-Project-1/assets/145722674/f559e623-e6b0-4a52-a90e-955d1c5e5b79)
![water regression](https://github.com/Cagan124/SMU-Project-1/assets/145722674/70464127-3a17-4eb5-8d14-a9ab3cebcf26)


	In conclusion, coffee/tea drinks are typically the most highly caffeinated, but energy drinks/shots have the most caffeine per 100 mL. Soda has significantly lower caffeine vs calorie levels on average. Water is low in both calories and caffeine on average. 
	In further studies, it may be more effective only to compare drinks from restaurants and coffee shops, or to compare personal-sized beverages from grocery stores. Further work could involve grouping brands to compare within a certain brand (e.g. which drink at Starbucks has the most caffeine?) or to compare between brands (e.g. is there more caffeine in a Starbucks iced coffee or a Dunkin iced coffee?). Furthermore, including prices would make this information even more valuable to consumers, as you could compare which drink gets you the best value if you are looking for more caffeine or more volume, for example.
