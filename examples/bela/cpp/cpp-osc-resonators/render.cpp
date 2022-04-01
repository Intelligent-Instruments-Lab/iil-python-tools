#include <stdlib.h>
#include <Bela.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include "ResonatorBank.h" // https://github.com/jarmitage/resonators

OscReceiver oscReceiver;
const char* remoteIp = "192.168.7.1";
const int localPort = 7562;

ResonatorBank resBank;
ResonatorBankOptions resBankOptions;
std::vector<ResonatorParams> resModelParams;
int resModelSize = 20;
bool resModelReceived = false;

float impulseInterval = 44100;
float impulseWidth = 0.1;
float impulseCount = 0;
bool audioInput = false;

void onReceive(oscpkt::Message* msg, const char* addr, void* arg) {

  if(msg->match("/resonators")) {
    
    printf("Model received...\n");
    auto argReader = msg->match("/resonators");
    for (int i = 0; i < resModelSize; i++) {
      argReader.popFloat(resModelParams[i].freq);
      argReader.popFloat(resModelParams[i].gain);
      argReader.popFloat(resModelParams[i].decay);
    }
    argReader.isOkNoMoreArgs();
    printf("\n");
    
    resBank.setBank(resModelParams);
    resBank.update();
    resModelReceived = true;

  } else {
    printf("Message address not recognised\n");
  }
}

bool setup (BelaContext *context, void *userData) {

  oscReceiver.setup(localPort, onReceive);
  
  resModelParams.resize(resModelSize);
  resBankOptions.total = resModelSize;
  resBank.setup(resBankOptions, context->audioSampleRate, context->audioFrames);

  return true;
}

void render (BelaContext *context, void *userData) { 

  if (resModelReceived) {

    for (unsigned int n = 0; n < context->audioFrames; ++n) {

      float in = 0.0f;

      if (audioInput) in = audioRead(context, n, 0); // an excitation signal
      else {
        if (++impulseCount >= impulseInterval * (1-impulseWidth))
          in = 1.0f * ( rand() / (float)RAND_MAX * 2.f - 1.f ); // noise
        if (impulseCount >= impulseInterval) impulseCount = 0;
      }
      float out = 0.0f;

      out = resBank.render(in);

      audioWrite(context, n, 0, out);
      audioWrite(context, n, 1, out);

    }
  
  }

}

void cleanup (BelaContext *context, void *userData) { }
