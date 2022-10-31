#include <Bela.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include <libraries/math_neon/math_neon.h>
#include <libraries/Scope/Scope.h>
#include <stdlib.h>
#include <time.h>

Scope scope;

#define NUM_OSCS 40

float gPhaseIncrement;

float gFrequencies[NUM_OSCS];
float gPhases[NUM_OSCS];

float gFrequenciesLFO[NUM_OSCS];
float gPhasesLFO[NUM_OSCS];

float gScale;

float gain = 5.0;

OscReceiver oscReceiver;
const char* remoteIp = "192.168.7.1";
const int localPort = 7562;

bool sinebankReceived = false;

void onReceive(oscpkt::Message* msg, void* arg) {

  if(msg->match("/sinebank")) {
    
    // printf("Model received...\n");
    auto argReader = msg->match("/sinebank");
    for (int i = 0; i < NUM_OSCS; i++) {
      argReader.popFloat(gFrequencies[i]);
      argReader.popFloat(gFrequenciesLFO[i]);
    }
    argReader.isOkNoMoreArgs();
    printf("\n");
    
    sinebankReceived = true;

  } else {
    printf("Message address not recognised\n");
  }
}

bool setup (BelaContext *context, void *userData) {

  scope.setup(3, context->audioSampleRate);

  oscReceiver.setup(localPort, onReceive);
  
  gPhaseIncrement = 2.0 * M_PI * 1.0 / context->audioSampleRate;
  gScale = 1 / (float)NUM_OSCS * 0.5;

  srand (time(NULL));

  for(int k = 0; k < NUM_OSCS; ++k){
      // Fill array gFrequencies[k] with random freq between 300 - 2700Hz
      gFrequencies[k] = rand() / (float)RAND_MAX * 2400 + 300;
      // Fill array gFrequenciesLFO[k] with random freq between 0.001 - 0.051Hz
      gFrequenciesLFO[k] = rand() / (float)RAND_MAX * 0.05 + 0.001;
      gPhasesLFO[k] = 0;
    }

  return true;
}

void render (BelaContext *context, void *userData) { 

  // if (sinebankReceived) {

    for(unsigned int n = 0; n < context->audioFrames; n++) {
      float out[2] = {0};

      for(int k = 0; k < NUM_OSCS; ++k){

        // Calculate the LFO amplitude
        float LFO = sinf_neon(gPhasesLFO[k]);
        gPhasesLFO[k] += gFrequenciesLFO[k] * gPhaseIncrement;
        if(gPhasesLFO[k] > M_PI)
          gPhasesLFO[k] -= 2.0f * (float)M_PI;

        // Calculate oscillator sinewaves and output them amplitude modulated
        // by LFO sinewave squared.
        // Outputs from the oscillators are summed in out[],
        // with even numbered oscillators going to the left channel out[0]
        // and odd numbered oscillators going to the right channel out[1]
        out[k&1] += sinf_neon(gPhases[k]) * gScale * (LFO*LFO) * gain;
        gPhases[k] += gFrequencies[k] * gPhaseIncrement;
        if(gPhases[k] > M_PI)
          gPhases[k] -= 2.0f * (float)M_PI;

      }
      // scope.log(out[0], out[1]);
      audioWrite(context, n, 0, out[0]);
      audioWrite(context, n, 1, out[1]);
    }
  
  // }

}

void cleanup (BelaContext *context, void *userData) { }
