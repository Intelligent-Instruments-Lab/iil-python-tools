#include <Bela.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include <cmath>

OscSender oscSender;
OscReceiver oscReceiver;

const char* remoteIp = "192.168.7.1";
const int remotePort = 7563;
const int localPort = 7562;

const int OSC_PACKET_LEN = 300;
const int OUT_CHANNELS = 2;

std::vector<float> oscInBuffer;
std::vector<float> oscOutBuffer;

float gIn1, gIn2;

int writePointer = 0;
int packetsSent = 0;
int gAudioFramesPerAnalogFrame = 0;

void onReceive(oscpkt::Message* msg, const char* addr, void* arg) {
  if(msg->match("/bela")) {
    auto argReader = msg->match("/bela");
    for (int i=0; i<OUT_CHANNELS * OSC_PACKET_LEN; i++)
      argReader.popFloat(oscInBuffer[i]);
    argReader.isOkNoMoreArgs();
  }
  printf("Printing oscInBuffer: ");
  for(auto f : oscInBuffer)
    printf("%f ", f);
  printf("\n");
}

bool setup(BelaContext *context, void *userData) {
  oscSender.setup(remotePort, remoteIp);
  oscReceiver.setup(localPort, on_receive);

  oscInBuffer.resize(OUT_CHANNELS * OSC_PACKET_LEN);
  oscOutBuffer.resize(OUT_CHANNELS * OSC_PACKET_LEN);
  
  if (context->analogFrames)
    gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;
    
  return true;
}

void render(BelaContext *context, void *userData) {
  for (unsigned int n = 0; n < context->audioFrames; ++n) {
    if (gAudioFramesPerAnalogFrame && !(n % gAudioFramesPerAnalogFrame)) {
      gIn1 = analogRead(context, n / gAudioFramesPerAnalogFrame, 0);
      gIn2 = analogRead(context, n / gAudioFramesPerAnalogFrame, 1);
    }
    
    oscOutBuffer[writePointer] = gIn1;
    oscOutBuffer[OSC_PACKET_LEN+writePointer] = gIn2;
    
    if (writePointer + 1 == OSC_PACKET_LEN) {
      oscSender.newMessage("/bela");
      oscSender.add(packetsSent);
      for (auto v : oscOutBuffer)
        oscSender.add(v);
      oscSender.send();
      packetsSent += 1;
    }
    writePointer = (writePointer+1)%OSC_PACKET_LEN;
  }
}

void cleanup(BelaContext*context, void *userData){}
