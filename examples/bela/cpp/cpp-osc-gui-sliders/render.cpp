/*
Based on \example Gui/sliders/render.cpp
*/

#include <Bela.h>
#include <libraries/Oscillator/Oscillator.h>
#include <libraries/Gui/Gui.h>
#include <libraries/GuiController/GuiController.h>
#include <libraries/OscSender/OscSender.h>
#include <libraries/OscReceiver/OscReceiver.h>
#include <cmath>

Gui gui;
GuiController controller;
Oscillator oscillator;

OscSender oscSender;
OscReceiver oscReceiver;
const char* remoteIp = "192.168.7.1";
const int localPort = 7562;
const int remotePort = 7563;

unsigned int gPitchSliderIdx;
unsigned int gAmplitudeSliderIdx;
float gPitch = 60.0;
float gAmplitude = 0.1;

void onReceive(oscpkt::Message* msg, const char* addr, void* arg) {

  if (msg->match("/pitch")) {

    float pitch;
    msg->match("/pitch").popFloat(pitch).isOkNoMoreArgs();
    if (gPitch != pitch) {
      gPitch = pitch;
      controller.setSliderValue(gPitchSliderIdx, gPitch);
      printf("Pitch %f\n", gPitch);
    }

  } else if (msg->match("/amplitude")) {

    float amplitude;
    msg->match("/amplitude").popFloat(amplitude).isOkNoMoreArgs();
    if (gAmplitude != amplitude) {
      gAmplitude = amplitude;
      controller.setSliderValue(gAmplitudeSliderIdx, gAmplitude);
      printf("Amplitude %f\n", gAmplitude);
    }

  } else {
    printf("Address not recognised\n");
  }

}

bool setup(BelaContext *context, void *userData)
{
  oscReceiver.setup(localPort, onReceive);
  oscSender.setup(remotePort, remoteIp);

  oscillator.setup(context->audioSampleRate);

  // Set up the GUI
  gui.setup(context->projectName);
  // and attach to it
  controller.setup(&gui, "Controls");

  // Arguments: name, default value, minimum, maximum, increment
  // store the return value to read from the slider later on
  gPitchSliderIdx = controller.addSlider("Pitch (MIDI note)", gPitch, 48, 84, 1); // step is 1: quantized semitones
  gAmplitudeSliderIdx = controller.addSlider("Amplitude", gAmplitude, 0, 0.5, 0.0001);
  return true;
}

void render(BelaContext *context, void *userData)
{
  // Access the sliders specifying the index we obtained when creating then
  float pitch = controller.getSliderValue(gPitchSliderIdx);
  float amplitude = controller.getSliderValue(gAmplitudeSliderIdx);
  if (gPitch != pitch) {
    gPitch = pitch;
    oscSender.newMessage("/pitch").add(gPitch).send();
  }
  if (gAmplitude != amplitude) {
    gAmplitude = amplitude;
    oscSender.newMessage("/amplitude").add(gAmplitude).send();
  }

  float frequency = 440 * powf(2, (gPitch-69)/12); // compute the frequency based on the MIDI pitch
  oscillator.setFrequency(frequency);
  // notice: no smoothing for amplitude and frequency, you will get clicks when the values change

  for(unsigned int n = 0; n < context->audioFrames; n++) {
    float out = oscillator.process() * gAmplitude;
    for(unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
      // Write the sample to every audio output channel
      audioWrite(context, n, channel, out);
    }
  }
}

void cleanup(BelaContext *context, void *userData)
{}
