import tkinter as tk
from tkinter import ttk ,filedialog
from threading import Thread
import subprocess
import sys
import os
import queue
output_queue =queue .Queue ()
process =None
params_dados ={
'a_F1':250 ,
'b_F1':543 ,
'a_F2':908 ,
'b_F2':1987 ,
'limiar_1':600 ,
'limiar_2':345 ,
'L':1 ,
'k_1':1 ,
'k_2':7 ,
'alvo_F1':421 ,
'alvo_F2':1887 ,
'neutro_F1':610 ,
'neutro_F2':1900
}
params_pesos ={
'lambda_zero':1.027 ,
'lambda_RP':0.417 ,
'lambda_RA':0.018
}
params_arq ={
'entrevistados':"1,3,5",
'vogais':"e"
}
name_mapping ={
'alvo_F1':'Target F1 [CA]',
'alvo_F2':'Target F2 [CA]',
'limiar_1':'Threshold (1) [CP]',
'limiar_2':'Threshold (2) [CP]',
'neutro_F1':'Neutral F1 [CA]',
'neutro_F2':'Neutral F2 [CA]',
'L':'L [CP]',
'k_1':'k (1) [CP]',
'k_2':'k (2) [CP]',
'a_F1':'F1 (min.)',
'b_F1':'F1 (max.)',
'a_F2':'F2 (min.)',
'b_F2':'F2 (max.)',
'lambda_zero':'Weight [N]',
'lambda_RA':'Weight [CA]',
'lambda_RP':'Weight [CP]',
'entrevistados':'Speakers',
'vogais':'Vowel'
}
def enqueue_output (output ,queue ):
    try :
        for line in iter (output .readline ,''):
            queue .put (line )
        output .close ()
    except Exception as e :
        queue .put ("Error reading output: "+str (e ))
def disable_inputs ():
    for key in entry_widgets :
        entry_widgets [key ].config (state ='disabled')
    optimize_checkbox .config (state ='disabled')
def enable_inputs ():
    for key in entry_widgets :
        entry_widgets [key ].config (state ='normal')
    optimize_checkbox .config (state ='normal')
    toggle_entries ()
if getattr (sys ,'frozen',False ):
    bundled_path =sys ._MEIPASS
    script_path =os .path .join (bundled_path ,'app.py')
else :
    script_path =os .path .abspath ('app.py')
def run_script ():
    global process
    disable_inputs ()
    modified_params ={key :entry_widgets [key ].get ()for key in entry_widgets }
    modified_params ['caminho_do_arquivo']=selected_file_path
    entrevistados_input =modified_params ['entrevistados'].replace (' ','')
    if entrevistados_input =="0":
        modified_params ['entrevistados']="todos"
    else :
        modified_params ['entrevistados']=','.join (entrevistados_input .split (','))
    modified_params ['vogais']=','.join (modified_params ['vogais'].split (', '))
    args =[f"--{key }={value }"for key ,value in modified_params .items ()]
    if optimize_var .get ():
        args .append ('--otimizar')
    print ("Parameters:",args )
    def execute ():
        global process
        script_path =os .path .abspath ('app.py')
        cmd =['python3','-u',script_path ]+args
        try :
            process =subprocess .Popen (cmd ,stdout =subprocess .PIPE ,stderr =subprocess .PIPE ,stdin =subprocess .PIPE ,text =True ,bufsize =1 ,universal_newlines =True )
            stdout_thread =Thread (target =enqueue_output ,args =(process .stdout ,output_queue ))
            stdout_thread .daemon =True
            stdout_thread .start ()
            process .wait ()
            stdout_thread .join ()
        except Exception as e :
            print ("Error:",e )
        root .after (0 ,lambda :run_button .config (state =tk .NORMAL ))
        root .after (0 ,enable_inputs )
    text_widget .config (state =tk .NORMAL )
    text_widget .delete ('1.0',tk .END )
    text_widget .insert (tk .END ,"Starting...\n")
    text_widget .config (state =tk .DISABLED )
    run_button .config (state =tk .DISABLED )
    stop_button .config (state =tk .NORMAL )
    thread =Thread (target =execute )
    thread .daemon =True
    thread .start ()
def stop_script ():
    global process
    if process and process .poll ()is None :
        process .terminate ()
        process =None
    text_widget .config (state =tk .NORMAL )
    text_widget .insert (tk .END ,"\nStopping...")
    text_widget .config (state =tk .DISABLED )
    run_button .config (state =tk .NORMAL )
    stop_button .config (state =tk .DISABLED )
    enable_inputs ()
def update_gui ():
    while not output_queue .empty ():
        line =output_queue .get_nowait ()
        if line :
            text_widget .config (state =tk .NORMAL )
            text_widget .insert (tk .END ,line )
            text_widget .config (state =tk .DISABLED )
            text_widget .see (tk .END )
    root .after (100 ,update_gui )
root =tk .Tk ()
root .title ("Continuous Violations Demonstration")
root .resizable (False ,False )
optimize_var =tk .BooleanVar (value =False )
def toggle_entries ():
    if optimize_var .get ():
        entry_widgets ['lambda_zero'].config (state ='disabled')
        entry_widgets ['lambda_RA'].config (state ='disabled')
        entry_widgets ['lambda_RP'].config (state ='disabled')
    else :
        entry_widgets ['lambda_zero'].config (state ='normal')
        entry_widgets ['lambda_RA'].config (state ='normal')
        entry_widgets ['lambda_RP'].config (state ='normal')
selected_file_path ='data.txt'
def select_file ():
    global selected_file_path
    filepath =filedialog .askopenfilename (
    title ="Select a file",
    filetypes =(("Text files","*.txt"),("All files","*.*"))
    )
    if filepath :
        selected_file_path =filepath
main_frame =ttk .Frame (root )
main_frame .pack (side =tk .LEFT ,fill =tk .BOTH ,expand =True )
side_frame =ttk .Frame (root ,padding ="10")
side_frame .pack (side =tk .RIGHT ,fill =tk .Y )
text_widget =tk .Text (main_frame ,height =31 ,width =100 )
text_widget .config (state =tk .DISABLED )
text_widget .pack (padx =10 ,pady =10 )
button_frame =ttk .Frame (main_frame )
button_frame .pack (pady =10 )
run_button =ttk .Button (button_frame ,text ="Start",command =run_script )
run_button .pack (side =tk .LEFT ,padx =5 )
stop_button =ttk .Button (button_frame ,text ="Stop",command =stop_script )
stop_button .pack (side =tk .LEFT ,padx =5 )
stop_button .config (state =tk .DISABLED )
select_file_button =ttk .Button (button_frame ,text ="Select Data",command =select_file )
select_file_button .pack (side =tk .LEFT ,padx =5 )
optimize_checkbox =ttk .Checkbutton (
side_frame ,
text ='Calculate weights?',
variable =optimize_var ,
command =toggle_entries ,
onvalue =True ,
offvalue =False
)
optimize_checkbox .pack (side =tk .TOP ,fill =tk .X ,padx =5 ,pady =3 )
entry_widgets ={}
label_width =14
entry_width =8
for i ,key in enumerate (list (params_dados )[:4 ]):
    row =ttk .Frame (side_frame )
    row .pack (side =tk .TOP ,fill =tk .X ,padx =5 ,pady =3 )
    label =ttk .Label (row ,text =name_mapping .get (key ,key )+":",width =label_width ,anchor ='e')
    label .pack (side =tk .LEFT )
    entry =ttk .Entry (row ,width =entry_width )
    entry .pack (side =tk .LEFT ,padx =5 )
    entry .insert (0 ,str (params_dados [key ]))
    entry_widgets [key ]=entry
ttk .Separator (side_frame ,orient ='horizontal').pack (side =tk .TOP ,fill ='x',pady =10 )
for key in params_arq :
    row =ttk .Frame (side_frame )
    row .pack (side =tk .TOP ,fill =tk .X ,padx =5 ,pady =3 )
    label =ttk .Label (row ,text =name_mapping .get (key ,key )+":",width =label_width ,anchor ='e')
    label .pack (side =tk .LEFT )
    entry =ttk .Entry (row ,width =entry_width )
    entry .pack (side =tk .LEFT ,padx =5 )
    initial_value =', '.join (str (x )for x in params_arq [key ].split (','))if isinstance (params_arq [key ],str )else str (params_arq [key ])
    entry .insert (0 ,initial_value )
    entry_widgets [key ]=entry
ttk .Separator (side_frame ,orient ='horizontal').pack (side =tk .TOP ,fill ='x',pady =10 )
for key in params_pesos :
    row =ttk .Frame (side_frame )
    row .pack (side =tk .TOP ,fill =tk .X ,padx =5 ,pady =3 )
    label =ttk .Label (row ,text =name_mapping .get (key ,key )+":",width =label_width ,anchor ='e')
    label .pack (side =tk .LEFT )
    entry =ttk .Entry (row ,width =entry_width )
    entry .pack (side =tk .LEFT ,padx =5 )
    entry .insert (0 ,str (params_pesos [key ]))
    entry_widgets [key ]=entry
ttk .Separator (side_frame ,orient ='horizontal').pack (side =tk .TOP ,fill ='x',pady =10 )
for key in list (params_dados )[4 :]:
    row =ttk .Frame (side_frame )
    row .pack (side =tk .TOP ,fill =tk .X ,padx =5 ,pady =3 )
    label =ttk .Label (row ,text =name_mapping .get (key ,key )+":",width =label_width ,anchor ='e')
    label .pack (side =tk .LEFT )
    entry =ttk .Entry (row ,width =entry_width )
    entry .pack (side =tk .LEFT ,padx =5 )
    entry .insert (0 ,str (params_dados [key ]))
    entry_widgets [key ]=entry
root .after (100 ,update_gui )
root .mainloop ()
