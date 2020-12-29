import App from './App.html';
import * as tf from '@tensorflow/tfjs';
import {shuffle, randomId, getQuery, getTurkInfo, setCookie, getCookie, Model, sinelinesExample, sinelinesLimits, sinelinesPcts, checkPermitted} from './utils';

window.tf = tf;

const rootUrl = location.pathname.split('/').slice(0,-1).join('/') + '/models';

// Persistent ID for the user (assuming they don't clear cookies)
const cookieId = 'dtakquizsession';
if (getCookie(cookieId)) {
  const userId = getCookie(cookieId);
  window.userId = userId;
} else {
  const userId = randomId();
  setCookie(cookieId, userId, 1);
  window.userId = userId;
}

// Ephemeral ID for this particular session
window.session = `142857_${randomId()}`;

let app = null;

async function load(dataKey, modelKeys) {
  const modelPromises = modelKeys.map(modelKey => {
    if (dataKey == 'sinelines' && modelKey == 'gt') {
      // In the special case of Sinelines + GT, we return a fake model
      // implemented in JavaScript, rather than a TensorFlow.js network
      return new Promise((resolve, reject) => {
        const model = new Model({
          decoder: sinelinesExample,
          dataKey: 'sinelines',
          modelKey: 'gt',
          z_lims: sinelinesLimits,
          z_pcts: sinelinesPcts,
          Dx: 64,
          Dz: 5,
          Dc: [],
          hierarchy: [
            { type: 'continuous' },
            { type: 'continuous' },
            { type: 'continuous' },
            { type: 'continuous' },
            { type: 'continuous' }
          ]
        });
        resolve(model);
      });
    } else {
      // Normally, we load a trained neural network with tf.loadFrozenModel, as
      // well as some configuration that lets us render the proper UI and
      // sample inputs
      return Promise.all([
        tf.loadFrozenModel(`${rootUrl}/${dataKey}/${modelKey}/decoder_web/tensorflowjs_model.pb`,
                           `${rootUrl}/${dataKey}/${modelKey}/decoder_web/weights_manifest.json`),
        fetch(`${rootUrl}/${dataKey}/${modelKey}/config.json`).then(async (resp) => await resp.json())
      ]).then(responses => {
        const decoder = responses[0];
        const config = responses[1];
        config.decoder = decoder;
        config.modelKey = modelKey;
        config.dataKey = dataKey;
        return new Model(config);
      });
    }
  });

  const viz = getQuery('viz') || 'sliders';

  // Figure out if we're on MTurk
  const wid = getTurkInfo().workerId;
  let embedded = false;
  if (getQuery('workerId')) {
    embedded = true;
  }

  // Figure out if the user has already completed this HIT
  let permitted;
  if (getQuery('permitted') || getQuery('skip_instructions')) {
    permitted = true;
  } else {
    permitted = await checkPermitted();
  }

  const appDiv = document.getElementById('app');

  if (permitted === false) {
    // If we detect that the user has already completed the task, and there
    // isn't a special query param saying that's alright, replace the loading
    // screen with a "not permitted" message
    appDiv.innerHTML = "<div class='warning-message'>It looks like you've already completed a similar HIT in the past! Unfortunately, for this study we require that each respondent be unique. <strong>Please return this HIT to avoid any impact on your approval rating.</strong> We apologize for any inconvenience or lack of clarity.</div>"
  } else {
    // Otherwise, replace the loading screen with our actual quiz application
    appDiv.innerHTML = '';
    app = new App({
      target: appDiv,
      data: { dataKey, modelKeys, viz, modelPromises, embedded, compcode: window.session }
    });
  }
}

const dataset = getQuery('dataset') || 'dsprites';
let models = getQuery('models');
if (models) {
  models = models.split('|');
} else {
  models = shuffle(['ae', 'vae', 'gt']);
}
load(dataset, models);

export default app;
