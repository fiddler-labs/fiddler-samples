"""
Fiddler python client.
"""
# TODO: Add License

from __future__ import absolute_import

from collections import namedtuple
import json
import requests


class FiddlerException(Exception):
    """An exception raised by client API in case of errors."""

    pass


API_BASE_URL = 'https://api.fiddler.ai'


# A PredictionEventBundle represents a batch of inferences and their input
# features. All of these share schema, latency, and success status. A bundle
# can consist just one event as well.
PredictionEventBundle = namedtuple(
    'PredictionEventBundle',
    [
        'prediction_status',  # type int # 0 for success, failure otherwise
        'prediction_latency',  # type float # Latency in seconds.
        'input_feature_bundle',  # list of feature vectors.
        'prediction_bundle',  # list of prediction vectors.
        # TODO: Support sending schema as well.
    ],
)


_protocol_version = 1


class Fiddler:
    """
    :param token: Token for Fiddler API (available under account settings
                   on https://fiddler.ai).
    :param organization: Name of the organization
    :param project: Project for the model
    :param model: Model name or similar identifier
    """

    def __init__(self, token, organization, project, model, base_url=API_BASE_URL):
        self.token = token
        self.organization = organization
        self.project = project
        self.model = model
        # TODO: Let user specify more metadata like version.
        self.base_url = base_url.rstrip('/')
        self._sender = _RequestSender()

    def _api_endpoint(self):
        return '{base}/external_event/{org}/{project}/{model}'.format(
            base=self.base_url,
            org=self.organization,
            project=self.project,
            model=self.model,
        )

    def publish_event(
        self, event, update_event=False, event_time_stamp=None, json_encoder=None
    ):
        """
        Publishes an event to Fiddler Service.
        :param dict event: Dictionary of event details, such as features
                           and predictions.
        :param update_event: if the event is an update to a previously
                             published row
        :param event_time_stamp: the UTC timestamp of the event in
                                 milliseconds
        :param json_encoder: Optional encoder for JSON encoding.
        :raises raises FiddlerException in case of a failure.
        """
        # More candidates for args: timestamp, or any specific details?

        if update_event:
            event['__event_type'] = 'update_event'
            if event_time_stamp:
                event['__updated_at'] = event_time_stamp
            if '__event_id' not in event.keys():
                raise ValueError('An update event needs an __event_id')
        else:
            event['__event_type'] = 'execution_event'
            if event_time_stamp:
                event['__occurred_at'] = event_time_stamp

        self._sender.send(self._api_endpoint(), self.token, event, json_encoder)


class _RequestSender:
    """
    Sends events to Fiddler API service and handles HTTP response and errors.
    """

    @staticmethod
    def send(api_endpoint, token, data, json_encoder=None):
        """
        Sends HTTP Post request to api_endpoint.
        :param api_endpoint: e.g. http://api.fiddler.com/
        :param token: token to send with the request.
        :param data: dictionary, which is sent as as JSON.
        :param json_encoder: Optional encoder for JSON encoding.
        :raises FiddlerException in case of
        errors or if the response from server does not indicate 'SUCCESS'.
        """

        # TODO: Either we should deprecate this script or we should add auth
        # and a way to forward the request to the right service. Right now
        # we are just making external_event requests which are in data_service.
        # If we start hitting executor_service endpoint, we might have to set
        # x-fdlr-fwd header appropriately.
        # More info in https://fiddlerlabs.atlassian.net/browse/FV-1081
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(token),
            'x-fdlr-fwd': 'data_service',
        }

        try:
            resp = requests.post(
                api_endpoint, data=json.dumps(data, cls=json_encoder), headers=headers
            )
        except requests.RequestException as e:
            raise FiddlerException(e)

        resp_json = None

        try:
            resp_json = resp.json()
        except ValueError:
            pass  # will throw right below to avoid nested exceptions.

        if not resp_json:
            # Server didn't send proper json response. It should.

            raise FiddlerException(
                'Error response from server: {}{}'.format(
                    resp.text[0:150], len(resp.text) > 150 and '...' or ''
                )
            )

        if resp_json.get('status') != 'SUCCESS':
            error_msg = resp_json.get('message', 'Unknown error from server')
            raise FiddlerException(error_msg)
