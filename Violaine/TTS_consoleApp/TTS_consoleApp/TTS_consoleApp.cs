using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Speech.Synthesis;

namespace TTS_consoleApp
{
    class TTS_consoleApp
    {
        static void Main(string[] args)
        {
            // Initialize a new instance of the SpeechSynthesizer.
            SpeechSynthesizer synth = new SpeechSynthesizer();

            // Configure the audio output. 
            synth.SetOutputToDefaultAudioDevice();

            foreach (InstalledVoice voice in synth.GetInstalledVoices())
            {
                VoiceInfo info = voice.VoiceInfo;

                Console.WriteLine(" Name:          " + info.Name);
                Console.WriteLine(" Culture:       " + info.Culture);
                Console.WriteLine(" Age:           " + info.Age);
                Console.WriteLine(" Gender:        " + info.Gender);
                Console.WriteLine(" Description:   " + info.Description);
                Console.WriteLine(" ID:            " + info.Id);
            }

            #region Different TTS engine & caracteristics
            /*---------------            
            Zira - Female - en US
            Microsoft Zira Desktop

            Heather - Female - en CA
            Microsoft Server Speech Text to Speech Voice (en-CA, Heather)

            Hazel - Female - en GB            
            Microsoft Server Speech Text to Speech Voice (en-GB, Hazel)

            Heera - Female - en IN
            Microsoft Server Speech Text to Speech Voice (en-IN, Heera)

            David - Male - en US
            Microsoft David Desktop
            
            Helen - Female - en US
            Microsoft Server Speech Text to Speech Voice (en-US, Helen)

            ZiraPro - Female - en US
            Microsoft Server Speech Text to Speech Voice (en-US, ZiraPro)

            Hayley - Female - en AU
            Microsoft Server Speech Text to Speech Voice (en-AU, Hayley)

            Hortense - Female - fr FR
            Microsoft Hortense Desktop            
            --------------- */
            #endregion

            PromptBuilder prompt = new PromptBuilder();
            //prompt.StartVoice(new System.Globalization.CultureInfo("en-US"));
            prompt.AppendText("Hi, I'm"); 
            synth.SelectVoice("Microsoft Server Speech Text to Speech Voice (en-US, ZiraPro)");
            string m_engineName = synth.Voice.Name + " !";
            prompt.AppendText(m_engineName);
            prompt.AppendText("How are you?");
            //prompt.AppendBreak();

            //synth.SelectVoice("Microsoft Zira Desktop");
            //synth.Speak(prompt);

            //synth.SelectVoice("Microsoft Server Speech Text to Speech Voice (en-US, Helen)");
            //synth.Speak(prompt);

            //synth.SelectVoice("Microsoft Server Speech Text to Speech Voice (en-AU, Hayley)");
            //synth.Speak(prompt);

            //synth.SelectVoice("Microsoft Server Speech Text to Speech Voice (en-CA, Heather)");
            //synth.Speak(prompt);

            //synth.SelectVoice("Microsoft Server Speech Text to Speech Voice (en-GB, Hazel)");
            //synth.Speak(prompt);

            //synth.SelectVoice("Microsoft Server Speech Text to Speech Voice (en-IN, Heera)");
            //synth.Speak(prompt);

            //synth.SelectVoice("Microsoft David Desktop");
            //synth.Speak(prompt);

            //synth.SelectVoice("Microsoft Hortense Desktop");
            //synth.Speak(prompt);

            

            //prompt.AppendSsmlMarkup("<prosody pitch=\"x-high\">");
            //prompt.AppendText("How are you?");
            //prompt.AppendSsmlMarkup("</prosody>");

            //prompt.AppendSsmlMarkup("<prosody contour="(0 %, +20Hz)(10 %, +30 %)(40 %, +10Hz)">");
            //prompt.AppendText("How are you?");
            //prompt.AppendSsmlMarkup("</prosody>");


            //prompt.AppendText("This is a long text. With two sentences! With two sentences.");

            //prompt.StartStyle(new PromptStyle(PromptVolume.Soft));
            //prompt.StartParagraph();
            //prompt.StartSentence();
            //prompt.AppendText("This is a long text. Dividing by sentence is completely useless.");
            //prompt.EndSentence();
            //prompt.EndParagraph();
            //prompt.EndStyle();

            //prompt.StartStyle(new PromptStyle(PromptRate.Fast));
            //prompt.AppendText("Please answer!");
            //prompt.EndStyle();

            MemoryStream audioStream = new MemoryStream();
            synth.SetOutputToWaveStream(audioStream);
            // Speak a string.
            synth.Speak(prompt);
            byte[] buffer = audioStream.ToArray();
            Console.WriteLine(buffer.Length);
            Console.WriteLine(audioStream.Length);

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}
