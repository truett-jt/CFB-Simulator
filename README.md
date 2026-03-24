All of our model testing is in cfb.ipynb. To be run the actual model used in our pipeline, see the classification.py file in the /models folder. Then follow these steps to be able to run the model:

1) Download the files from this link:
[CFB-Simulator](https://gtvault-my.sharepoint.com/:f:/g/personal/jwehner7_gatech_edu/IgC3vYAwk2plQ7mTpbhgliL_ARPqxtMFDZOrsckxGfVYrWs?e=n96iGu)

2) Copy the files to the following directory where ever you cloned the repo: *CFB-Simulator/data/*

You can also use cfb.ipynb to make the API calls that will load the necessary data to csv files in the /data folder. To do so, follow these steps:

1) Copy the .env.example file and rename it ".env".
2) Generate a new API Key here:
3) Place API key in the .env folder. This key will not be visible to anyone on Github.
4) Run the "Data Collection" cells in cfb.ipynb
