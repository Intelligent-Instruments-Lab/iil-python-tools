#include <stdlib.h>
#include <Bela.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include "Resonator.h" // https://github.com/jarmitage/resonators

OscReceiver oscReceiver;
const char* remoteIp = "192.168.7.1";
const int localPort = 7562;

Resonator res;
ResonatorOptions options; // will initialise to default

bool audioInput = false;
float impulseInterval = 44100;
float impulseWidth = 0.125;
float impulseCount = 0;

void onReceive(oscpkt::Message* msg, const char* addr, void* arg) {

  if(msg->match("/resonator")) {
    float freq, gain, decay;
    msg->match("/resonator")
      .popFloat(freq)
      .popFloat(gain)
      .popFloat(decay)
      .isOkNoMoreArgs();
    res.setParameters(freq, gain, decay);
    res.update();
    printf("Freq: %f, Gain: %f, Decay: %f\n", freq, gain, decay);

  } else if (msg->match("/resonator/freq")) {

    float freq;
    msg->match("/resonator/freq").popFloat(freq).isOkNoMoreArgs();
    res.setParameter(0, freq);
    res.update();
    printf("Freq: %f\n", freq);

  } else if (msg->match("/resonator/gain")) {

    float gain;
    msg->match("/resonator/gain").popFloat(gain).isOkNoMoreArgs();
    res.setParameter(1, gain);
    res.update();
    printf("Gain: %f\n", gain);

  } else if (msg->match("/resonator/decay")) {

    float decay;
    msg->match("/resonator/decay").popFloat(decay).isOkNoMoreArgs();
    res.setParameter(2, decay);
    res.update();
    printf("Decay: %f\n", decay);

  } else {
    printf("Message address not recognised\n");
  }
}

bool setup (BelaContext *context, void *userData) {

  oscReceiver.setup(localPort, onReceive);

  res.setup(options, context->audioSampleRate, context->audioFrames);
  res.setParameters(440, 0.1, 0.5); // freq, gain, decay
  res.update(); // update the state of the resonator based on the new parameters

  return true;
}

void render (BelaContext *context, void *userData) { 

  for (unsigned int n = 0; n < context->audioFrames; ++n) {

    float in = 0.0f;
    if (audioInput) in = audioRead(context, n, 0); // an excitation signal
    else {
      if (++impulseCount >= impulseInterval * (1-impulseWidth))
        in = 1.0f * ( rand() / (float)RAND_MAX * 2.f - 1.f ); // noise
      if (impulseCount >= impulseInterval) impulseCount = 0;
    }
    float out = 0.0f;

    out = res.render(in);

    audioWrite(context, n, 0, out);
    audioWrite(context, n, 1, out);
  
  }

}

void cleanup (BelaContext *context, void *userData) { }
