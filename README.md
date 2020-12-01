<a name="quick-start-guide"></a>

<div align="left">
    <img src="https://global-uploads.webflow.com/5e067beb4c88a64e31622d4b/5efa291bd80756354b0968a9_fiddler-logo-p-500.png"
         alt="Image of Fiddler logo"/>
</div>

# Quick Start Guide

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
   
6. Configure auth token:

   Login to fiddler account and copy auth token from Settings > Credentials > Key

   Edit `content_root/tutorial/fiddler.ini`

```
   [FIDDLER]
   url = https://<your-org-cluster>.fiddler.ai
   org_id = <your-org-account>
   auth_token = <your-auth-token>
```

<a name="step-by-step"></a>
# Step By Step Tutorial:

- [How to monitor data](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/01%20Basic%20model%20monitoring.ipynb)
- [How to monitor with surrogate explanation](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/01a%20Model%20monitoring%20with%20surrogate%20explanation.ipynb)
- [How to import sklearn regression model](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02%20How%20to%20upload%20a%20simple%20sklearn%20regression%20model.ipynb)
- [Debug model import problems](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/03%20How%20to%20debug%20model%20upload.ipynb)
- [How to generate a model from your tabular data](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/04%20automodel.ipynb)
- [Publish realtime events to Fiddler](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/06%20publish_event.ipynb)
- [How to import a model hosted on an external server](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/05%20Import%20model%20hosted%20outside%20of%20Fiddler.ipynb)
- [Explaining model hosted on Sagemaker](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/09%20Importing%20model%20hosted%20on%20Sagemaker.ipynb)
- [How to upload a keras model using tabular data with IG enabled](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/07%20How%20to%20upload%20a%20keras%20model%20using%20tabular%20data%20with%20IG%20enabled.ipynb)
- [How to upload a tf model using text data with IG enabled](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/08%20How%20to%20upload%20a%20tf%20model%20using%20text%20data%20with%20IG%20enabled.ipynb)

   
<a name="license"></a>
# License

```
See LICENSE File for details. 
```

<a name="i-want-to-know-more"></a>
# I want to know more!

Here are some links that you will find useful:
* **[Documentation](https://docs.fiddler.ai/)**
* **[todo: Video tutorial](https://fiddler.ai)**
* **[Full API Reference](https://docs.fiddler.ai/api-reference/python-package/)**
* **[Fiddler Slack](https://fiddler-community.slack.com/)**


<a name="want-to-contribute"></a>
# Want to Contribute?

This is an open source project, and we'd love to see your contributions!
Please git clone this project and send us a pull request. Thanks.




   
   
