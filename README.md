<a name="getting-started"></a>

<div align="left">
    <img src="content_root/tutorial/images/logo.png"
         alt="Image of Fiddler logo"/>
</div>

# Getting Started

This package contains tutorials, samples, and documentation for using Fiddler.

1. Clone this repo to your local machine:

   ```git clone https://github.com/fiddler-labs/fiddler-samples.git```

2. Build notebook server
   
   Prerequisite: 
   
      Docker - min. Docker EE 18.09 or CE 17.12.1 installed and running.

   ```cd fiddler-samples; make build``` 

3. Start noteboook server

   ```make run```

4. Notebook service is now running at http://localhost:7100

5. To try out the tutorial you will also need Fiddler server. You can either get a cloud account or download Fiddler Onebox from a link that will be emailed to you
   
6. Configure fiddler client:

   Login to fiddler account and copy auth token from Settings > Credentials > Key
   
   To update token and client URL visit: 

   content_root/tutorial/00 Install & Setup.ipynb

```
   [FIDDLER]
   url = https://<your-org-cluster>.fiddler.ai
   org_id = <your-org-account>
   auth_token = <your-auth-token>
```

<a name="examples"></a>
# Examples:
*The goal of these notebooks is to show you how to walkthrough installation, monitoring, and model upload of Fiddler using different model frameworks and data types. You can also use these as a reference guide to upload your dataset and model along with production traffic that you want to monitor, into Fiddler.*

#### Installation
- [Install & Setup](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/00%20Install%20%26%20Setup.ipynb)
#### Monitoring
- [Monitoring Quick Start](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/01a%20Monitoring%20Quickstart.ipynb)
- [Monitoring With Your Model](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/01b%20Monitoring%20With%20Your%20Model%20.ipynb)
#### Model Upload
- [Sklearn Tabular Model](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02a%20Sklearn%20Tabular%20Model.ipynb)
- [Tensorflow Tabular Model](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02b%20Tensorflow%20Tabular%20Model.ipynb)
- [Tensorflow Text Model](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02c%20Tensorflow%20Text%20Model.ipynb)
- [PyTorch Tabular Model](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02d%20PyTorch%20Tabular%20Model.ipynb)
- [Model Upload Using Containers](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02e%20Model%20Upload%20Using%20Containers.ipynb)
- [Debug Model Upload Issues](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02f%20Debug%20Model%20Upload%20Issues.ipynb)

   
<a name="license"></a>
# License

```
See LICENSE File for details. 
```

<a name="i-want-to-know-more"></a>
# I want to know more!

Here are some links that you will find useful:
* **[Documentation](https://docs.fiddler.ai/)**
* **[Full API Reference](https://docs.fiddler.ai/api-reference/python-package/)**
* **[Fiddler Slack](https://fiddler-community.slack.com/)**


<a name="want-to-contribute"></a>
# Want to Contribute?

This is an open source project, and we'd love to see your contributions!
Please git clone this project and send us a pull request. Thanks.




   
   
