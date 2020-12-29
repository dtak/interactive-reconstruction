import {Store} from 'svelte/store.js';

export function range(n,n2) {
  const rng = [];
  if (n2) {
    for (let i = n; i < n2; i++) {
      rng.push(i);
    }
  } else {
    for (let i = 0; i < n; i++) {
      rng.push(i);
    }
  }
  return rng;
}

export function mse(x1, x2) {
  let res = 0;
  for (let i = 0; i < x1.length; i++) {
    res += Math.pow(x1[i] - x2[i], 2);
  }
  return res;
}

export function pctAgreement(x1, x2, tolerance) {
  let tol = tolerance || 0.01;
  let same = 0;
  for (let i = 0; i < x1.length; i++)
    same += Math.abs(x1[i] - x2[i]) < tol;
  return same / x1.length;
}

export function bwIoU(x1, x2) {
  let numer = 0;
  let denom = 0;
  for (let i = 0; i < x1.length; i++) {
    const a = Math.round(x1[i]);
    const b = Math.round(x2[i]);
    numer += a && b;
    denom += a || b;
  }
  if (denom) {
    return numer / denom;
  } else {
    return 1;
  }
}

export function scaledMAE(x1, x2) {
  let num = 0;
  let den = 0.01;
  for (let i = 0; i < x1.length; i++) {
    den += Math.abs(x1[i]);
    den += Math.abs(x2[i]);
    num += Math.abs(x1[i] - x2[i]);
  }
  return num / den;
}

export function randomId() {
  return `${new Date().getTime()}_${Math.random().toString().slice(2)}`;
}

export function copyArray(a) {
  return JSON.parse(JSON.stringify(a));
}

export function sampleOne(array) {
  return array[Math.floor(Math.random()*array.length)];
}

export function getQuery(variable) {
  var query = window.location.search.substring(1);
  var vars = query.split('&');
  for (var i = 0; i < vars.length; i++) {
    var pair = vars[i].split('=');
    if (decodeURIComponent(pair[0]) == variable) {
      return decodeURIComponent(pair[1]);
    }
  }
  return null;
}

export function shuffle(arr) {
  var shuffled = arr.slice(0), i = arr.length, temp, index;
  while (i--) {
    index = Math.floor((i + 1) * Math.random());
    temp = shuffled[index];
    shuffled[index] = shuffled[i];
    shuffled[i] = temp;
  }
  return shuffled;
}

export function setCookie(cname, cvalue, exdays) {
  var d = new Date();
  d.setTime(d.getTime() + (exdays*24*60*60*1000));
  var expires = "expires="+ d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

export function getCookie(cname) {
  var name = cname + "=";
  var decodedCookie = decodeURIComponent(document.cookie);
  var ca = decodedCookie.split(';');
  for(var i = 0; i <ca.length; i++) {
    var c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

export function getTurkInfo() {
  const turk = {};
  const param = function(url, name) {
    name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
    var regexS = "[\\?&]" + name + "=([^&#]*)";
    var regex = new RegExp(regexS);
    var results = regex.exec(url);
    return (results == null) ? "" : results[1];
  };

  const src = param(window.location.href, "assignmentId") ? window.location.href : document.referrer;
  const keys = ["assignmentId", "hitId", "workerId", "turkSubmitTo"];
  keys.map((key) => {
    turk[key] = unescape(param(src, key));
  });

  turk.previewMode = (turk.assignmentId == "ASSIGNMENT_ID_NOT_AVAILABLE");
  turk.outsideTurk = (!turk.previewMode && turk.hitId === "" && turk.assignmentId == "" && turk.workerId == "")
  return turk;
};

export function submitToTurk(extraData) {
  const turkInfo = getTurkInfo();
  const assignmentId = turkInfo.assignmentId;
  const turkSubmitTo = turkInfo.turkSubmitTo;
  if (!assignmentId || !turkSubmitTo) return;
  const dataString = [];
  const data = extraData || {};
  for (const key in data) {
    if (data.hasOwnProperty(key)) {
      dataString.push(key + "=" + escape(data[key]));
    }
  }
  dataString.push("assignmentId=" + assignmentId);
  const url = turkSubmitTo + "/mturk/externalSubmit?" + dataString.join("&");
  window.location.href = url;
};

export const sinelinesX = [
  -3.2, -3.1, -3. , -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2,
  -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1,
  -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.0,
   0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,  1.1,
   1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2,
   2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1
];

export const sinelinesLimits = [
  [-1, 1],
  [-4, 4],
  [0, 10],
  [0, 10],
  [0, Math.round(2*Math.PI*100)/100]
];

export const sinelinesPcts = {
  0: [-0.999893135913239, -4.0281563225185515, 8.961289244786438e-06, 5.30015059428693e-05, 0.00016322766089180458],
  1: [-0.980001754475396, -2.311269540748075, 0.009841987035866726, 0.010443744378365831, 0.06186736006541578],
  2: [-0.9591688695071544, -2.0597147184340754, 0.020172961922099046, 0.019924062418860723, 0.12638822768553348],
  5: [-0.8994738404725823, -1.6427975809696242, 0.05129308348820467, 0.05070599647225438, 0.32238116731493033],
  25: [-0.4941112717424068, -0.6686323453876338, 0.2866509430647027, 0.2883639438105464, 1.5761996757550798],
  33: [-0.33375182815617055, -0.4373575008559303, 0.39893723386147084, 0.40118439436802233, 2.0825400887507897],
  45: [-0.09156294724735049, -0.12307848967091312, 0.5954192268013553, 0.5949511401031627, 2.836016251220051],
  48: [-0.03376775734792464, -0.046741919847168065, 0.6515293335680905, 0.65199031145571, 3.025931706046867],
  50: [0.006909467659452551, 0.0030745765028161558, 0.6893712424079417, 0.692993909968308, 3.1471703713066974],
  52: [0.04549216284549388, 0.055686621517758286, 0.7297143912257275, 0.7353312031774573, 3.268321217071762],
  55: [0.10550435593921875, 0.134326116379127, 0.7939679561188211, 0.8010356603962991, 3.4562392792396857],
  67: [0.34094280794872633, 0.45290684225955835, 1.1047393275041903, 1.1099313405200761, 4.20633149527023],
  75: [0.4998243442458409, 0.6855147522702414, 1.3778194991448511, 1.3861515588947089, 4.696628401872491],
  95: [0.8976675506469572, 1.657334488497137, 2.9811892605869037, 2.976451862795523, 5.954853628563546],
  98: [0.9588376947631542, 2.0786385449381495, 3.867735015756658, 3.928039182143862, 6.146009016541035],
  99: [0.9787371305578826, 2.3586180822663976, 4.499254127504022, 4.545999088089283, 6.2125160421037435],
  100: [0.9999663855650138, 3.909037947948695, 12.033450628308787, 11.29280329076291, 6.282994642616362]
};

export function sinelinesExample(z) {
  return sinelinesX.map((x) => {
    return x*z[0] + z[1] + z[2] * Math.sin(x*z[3] + z[4]);
  });
}

export function Unif(a,b) {
  return a + (b-a)*Uniform();
}

export function Uniform() {
  return tf.randomUniform([1]).dataSync()[0];
}
export function Normal() {
  return tf.randomNormal([1]).dataSync()[0];
}
export function Exponential() {
  return -Math.log(Uniform());
}

export function sampleSinelines() {
  return tf.tidy(() => {
    const m = 2 * (Uniform() - 0.5);
    const b = Normal();
    const A = Exponential();
    const w = Exponential();
    const ph = 2 * Math.PI * Uniform();
    return [m,b,A,w,ph].map((el,i) => {
      const rounded = Math.round(el*100)/100;
      const clamped = Math.min(
        Math.max(
          rounded,
          sinelinesLimits[i][0]
        ),
        sinelinesLimits[i][1]
      );
      return clamped;
    });
  });
};

export function toy2DExample(z) {
  const res = [];
  const color = 1 - z[1];
  for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 64; j++) {
      const inside = (Math.sqrt(Math.pow(i-31, 2) + Math.pow(j-31, 2)) < 24 * z[0] + 4);
      res.push(inside * color + (1-inside) * (1-color));
    }
  }
  return res
}

export function setupDimensionMetadata(data) {
  if (data.z && data.z.length) {
    if (typeof data.Dx == 'undefined') {
      if (data.dataKey == 'sinelines')
        data.Dx = 64;
      else if (data.dataKey == 'dsprites')
        data.Dx = 64*64;
    }
    if (typeof data.Dz == 'undefined') {
      data.Dz = data.z[0].length;
    }
    if (typeof data.Dc == 'undefined') {
      data.Dc = data.c.map((c,i) => {
        return c[0].length;
      });
    }
  } else {
    data.numExamples = [];
    data.c = [];
    data.z = [];
  }
  data.Dcz = data.Dz;
  data.Dc.forEach(d => data.Dcz += d);

  let k = 0;

  data.discDims = data.Dc.map((numOpts,i) => {
    const res = {
      id: `disc-dim-${i}`,
      name: `Dial #${i+1}`,
      type: 'discrete',
      discIndex: i,
      index: k,
      options: range(numOpts).map((j) => {
        return {
          id: `disc-dim-${i}-val-${j}`,
          name: `Value ${j+1}`,
          optIndex: j,
          index: k+j
        };
      })
    };
    k += numOpts;
    return res;
  });

  data.contDims = range(data.Dz).map((i) => {
    return {
      id: `cont-dim-${i}`,
      name: `Dial #${i+1+data.Dc.length}`,
      type: 'continuous',
      min: data.z_lims[i][0],
      max: data.z_lims[i][1],
      contIndex: i,
      index: k+i,
    }
  });

  data.flatDims = data.discDims.concat(data.contDims);

  window.data_ = data;
}

export class Model extends Store {
  constructor(data) {
    setupDimensionMetadata(data);

    const { Dc, Dz, discDims, contDims, z_lims } = data;

    if (data.z && data.z.length) {
      window.samp_order = shuffle(range(data.z.length));
      window.samp_idx = 0;
    }

    data.sampleLatentComponents = () => {
      if (data.z && data.z.length) {
        // If given, sample from a deterministic list of latent values
        const i = window.samp_order[window.samp_idx % data.z.length];
        window.samp_idx = window.samp_idx + 1;
        const cs = [];
        const z = [];
        for (let val of data.z[i]) {
          z.push(Math.round(100 * val) / 100); // IMPORTANT: round for sliders
        }
        for (let j of range(Dc.length)) {
          cs.push(copyArray(data.c[j][i]));
        }
        return { cs, z };
      } else if (data.dataKey == 'sinelines' && data.modelKey == 'gt') {
        const z = sampleSinelines();
        const cs = [];
        return { cs, z };
      } else {
        // Otherwise, generate latent values from discrete or continuous uniforms
        const cs = [];
        const z = [];
        for (let dc of Dc) {
          const c = range(dc).map((_) => 0);
          c[sampleOne(range(dc))] = 1;
          cs.push(c);
        }
        for (let i of range(Dz)) {
          const lo = z_lims[i][0];
          const hi = z_lims[i][1];
          const val = lo + (hi-lo) * Math.random();
          z.push(Math.round(100 * val) / 100);
        }
        return { cs, z };
      }
    }

    data.mergeCMZ = (cs, z) => {
      let res = [];
      for (let c of cs)
        res = res.concat(c);
      res = res.concat(z);
      return res;
    }

    data.sampleCMZ = () => {
      const { cs, z } = data.sampleLatentComponents();
      return data.mergeCMZ(cs, z);
    }

    if (typeof data.decoder.predict == 'undefined') {
      data.decode = data.decoder;
      data.decodes = (cmzs) => cmzs.map(data.decoder);
    } else {
      data.decode = (cmz) => {
        return tf.tidy(() => {
          const tensor = tf.tensor2d([cmz]);
          return data.decoder.predict(tensor).dataSync();
        });
      }

      data.decodes = (cmzs) => {
        if (!cmzs.length) return [];
        const output = tf.tidy(() => {
          const tensor = tf.tensor2d(cmzs);
          return data.decoder.predict(tensor).dataSync();
        });
        return range(cmzs.length).map((k) => {
          return output.slice(k*data.Dx, (k+1)*data.Dx);
        });
      }
    }

    data.getLoMedHi = (dim) => {
      // Sample "low", "medium", and "high" values from uniform distributions
      // over the 1st-5th percentiles, 48th-52nd percentiles, and 95th-99th
      // percentiles of the empirical distribution, respectively
      const lo = Unif(data.z_pcts[1][i], data.z_pcts[5][i]);
      const hi = Unif(data.z_pcts[95][i], data.z_pcts[99][i]);
      const mid = Unif(data.z_pcts[48][i], data.z_pcts[52][i]);
      return [lo, mid, hi];
    }

    super(data);
  }
}

export function getWorkerId() {
  let workerId = getTurkInfo().workerId;
  if (!workerId) {
    if (getCookie('dtakquizworker')) {
      workerId = getCookie('dtakquizworker');
    } else {
      workerId = window.userId;
    }
  } else {
    setCookie('dtakquizworker', workerId, 365);
  }
  return workerId;
}

export async function checkPermitted() {
  let checkUrl;
  if (location.hostname == 'localhost')
    checkUrl = 'http://localhost:3000/completions';
  else
    checkUrl = 'https://dtak-mturk-survey-data.herokuapp.com/completions';

  const workerId = getWorkerId();
  const scope = 'latent-rep-interpretability-quiz';

  checkUrl = `${checkUrl}?worker_id=${workerId}&scope=${scope}`;

  const response = await fetch(checkUrl);
  const json = await response.json();

  if (json && json.status == 'first_time') {
    return true;
  } else if (json && json.status == 'completed') {
    return false;
  } else {
    return undefined;
  }
}

export function logEvent(event, context) {
  let logUrl;
  if (location.hostname == 'localhost')
    logUrl = 'http://localhost:3000/logs';
  else
    logUrl = 'https://dtak-mturk-survey-data.herokuapp.com/logs';

  const body = {
    scope: 'latent-rep-interpretability-quiz',
    key: event.name,
    value: {
      userId: window.userId,
      workerId: getWorkerId(),
      session: window.session,
      timestamp: new Date().getTime(),
      eventData: event.data,
      eventContext: context,
      turkInfo: getTurkInfo()
    }
  };

  if (location.hostname == 'localhost') {
    console.log(body);
  }

  fetch(logUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
}
