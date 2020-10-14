<a name="quick-start-guide"></a>
# Quick Start Guide

This package contains tutorials, samples and documentation for using Fiddler.

1. Clone this repo to your local machine:

   ```git clone https://github.com/fiddler-labs/Fiddler-Tutorials.git```

2. Build notebook server

   ```cd fiddler-samples; make build``` 

3. Start noteboook server

   ```make run```

4. Notebook service is now running at http://localhost:7100

5. To try out the tutorial you will also need Fiddler server. You can either get a cloud account or download Fiddler image from https://fiddler.ai 
   
6. Configure auth token
Login to fiddler account and copy auth token from Settings > Credentials > Key

Edit content_root/tutorial/fiddler.ini

```
[FIDDLER]
url = https://<your-org>.fiddler.ai
org_id = <your-org>
auth_token = <your-auth-token>
```

<a name="step-by-step"></a>
# Step By Step Tutorial:

- [How to Monitor data?](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/01%20Monitor%20data%20using%20Fiddler.ipynb)
- [How to import sklearn regression model?](https://github.com/fiddler-labs/fiddler-samples/blob/master/content_root/tutorial/02_1%20How%20to%20upload%20a%20simple%20sklearn%20regression%20model.ipynb)
- 

   
   
   
<a name="license"></a>
# License

```
See LICENSE File for details. 
```

<a name="i-want-to-know-more"></a>
# I want to know more!

Here are some links that you will find useful:
* **[todo: Documentation](https://fiddler.ai)**
* **[todo: Video tutorial](https://fiddler.ai)**
* **[todo: Full API Reference](https://fiddler.ai)**
* **[todo: Fiddler slack](https://fiddler.ai)**


<a name="want-to-contribute"></a>
# Want to Contribute?

This is an open source project, and we'd love to see your contributions!
Please git clone this project and send us a pull request. Thanks.




   
   
