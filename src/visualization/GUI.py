import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button,LassoSelector
from matplotlib.path import Path
from ..core.mode_solver import calc_n_eff_ML
from ..core.mode_solver import calc_n_eff,calc_n_eff_general
from ..utils.help_functs import dispersion_calc,reshape_n_eff,segment_data
# import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import seaborn as sns





class LassoManager:
    def __init__(self, ax,ax2,ax3, data):
        self.ax = ax
        self.ax2 = ax2
        self.ax3 = ax3
        self.canvas = ax.figure.canvas
        self.data = data
        self.add_data = False
        # Initial scatter plot
        self.scatter = ax.scatter(data[:, 0], data[:, 1], color='k', s=5,edgecolors='none',label='MOLINO')
        self.scatter2 = ax2.scatter([],[],color='k' ,s=5,edgecolors='none')
        self.scatter3 = ax3.scatter([],[],color='k' ,s=5,edgecolors='none')
        # Lasso selector attributes
        self.lasso_selector = LassoSelector(ax, onselect=self.on_select)
        self.selected_index = np.zeros(len(data), dtype=bool)

        # Add button - the button axes must be in figure-relative coordinates
        self.button_ax = self.ax.figure.add_axes([0.0, 0.0, 0.1, 0.05])
        self.button = Button(self.button_ax, 'Add', color='lightgray')
        self.button.on_clicked(self.add_to_selection)
        
        

    def on_select(self, verts):
        path = Path(verts)
        # Update selected_index based on lasso selection
        currently_selected = path.contains_points(self.data)
        if self.add_data:
            self.selected_index = np.logical_or(self.selected_index, currently_selected)
        else:
            self.selected_index =  currently_selected

        self.update_plot_colors()
        dispersion_data = np.array(self.data[self.selected_index])
        xarrays,yarrays = segment_data(dispersion_data[:,0],dispersion_data[:,1])
        # vg_tot_sel = []
        # gvd_tot_sel = []
        # try:
        #    lines2.pop(0).remove()
        #    lines3.pop(0).remove()
        # except:
        #     pass
        lines2 = []
        lines3 = []
        for ii in range(len(xarrays)):            
            x_data = np.array(xarrays[ii])
            y_data = np.array(yarrays[ii])
            vp_select, vg_select, gvd_select  = dispersion_calc(x_data,y_data)
            lines2.append(self.ax2.plot(x_data,vg_select))
            lines3.append(self.ax3.plot(x_data,gvd_select))
        #     vg_tot_sel.append(vg_select)
        #     gvd_tot_sel.append(gvd_select)
        # self.scatter2.set_offsets(list( zip(xarrays,vg_tot_sel)))
        # self.scatter3.set_offsets(list(zip(xarrays,gvd_tot_sel)))

    def add_to_selection(self, event):
        if self.add_data:
            self.add_data = False
            self.button.label.set_text("Add")
        else:
            self.add_data = True
            self.button.label.set_text("Select")
       

    def update_plot_colors(self):
        # Update scatter plot colors based on the selection
        colors = ['red' if sel else 'k' for sel in self.selected_index]
        self.scatter.set_facecolors(colors)
        self.canvas.draw_idle()
        
    def update_data(self, new_data):
        # Update the data with new points
        self.data = new_data
        self.selected_index = np.zeros(len(new_data), dtype=bool)  # Reset selection

        # Update the scatter plot with the new data
        self.scatter.set_offsets(new_data)
        self.update_plot_colors()

def run_gui_general(lambda_vector, n_CORE, n_SUB, n_list_CLAD, w_CORE, w_list_CLAD, m):

    n_eff = calc_n_eff_general(lambda_vector, n_CORE,
                     n_SUB, n_list_CLAD, w_CORE, m, w_list_CLAD)
    n_eff_arr = reshape_n_eff(lambda_vector,n_eff)
    ### calculate the simple version of the asymmtric thick cladding
    n_eff_simple = calc_n_eff_general(
        lambda_vector, n_CORE, n_SUB, n_list_CLAD[:,0], w_CORE, m)
    n_eff_smp_list = [item for sublist in n_eff_simple for item in sublist]
    ### calculate vg, betta2
    vps,  vgs, betta2s  = dispersion_calc(lambda_vector,n_eff_smp_list)
    
    
    ### make figure
    fig, ax = plt.subplots(2, 2)#, layout='constrained')
    sns.set_theme()

    ###############   AX [0,0]   ###############
    l = ['MOLINO', 'Simple', 'Core index']

    lasso_var = LassoManager(ax[0,0],ax[1,0],ax[1,1], n_eff_arr)
    
    line2, = ax[0,0].plot(lambda_vector,  n_eff_simple, label=l[1])
    line3, = ax[0,0].plot(lambda_vector,  n_CORE, label=l[2], color='black', linestyle='--')
    ax[0,0].set_ylabel('Refractive index')
    ax[0,0].set_xlabel('Wavelength [$\mu$m]')
    ax[0,0].set_xlim(lambda_vector[0], lambda_vector[-1])
    ax[0,0].legend()
    
    ###############   AX [0,1]   ###############
    ax[0,1].text(0.15, 0.5, '<- Select region', dict(size=15))
    # ax[0,1].set_ylabel('Difference')
    # ax[0,1].set_xlabel('Wavelength [$\mu$m]')
    # ax[0,1].set_xlim(lambda_vector[0], lambda_vector[-1])
    # formatter = ScalarFormatter(useOffset=True) # This will use scientific notation
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1)) 
    # ax[0,1].yaxis.set_major_formatter(formatter)

    ###############   AX [1,0]   ###############
    
    # line_vg1, = ax[1,0].plot(lambda_vector,  vg)
    line_vg2, = ax[1,0].plot(lambda_vector,  vgs)
    ax[1,0].set_ylabel('Group velocity')
    ax[1,0].set_xlabel('Wavelength [$\mu$m]')
    ax[1,0].set_xlim(lambda_vector[0], lambda_vector[-1])
    
    # ###############   AX [1,1]   ###############
    
    # line_disp1, = ax[1,1].plot(lambda_vector,  betta2)
    line_disp2, = ax[1,1].plot(lambda_vector,  betta2s)
    ax[1,1].set_ylabel('Group velocity dispersion')
    ax[1,1].set_xlabel('Wavelength [$\mu$m]')
    ax[1,1].set_xlim(lambda_vector[1], lambda_vector[-2])
    # ax[1,1].set_ylim(-500,2000)
    
    
    ############### Define the sliders ###############
    ax_a = plt.axes([0.18, 0.15, 0.25, 0.03])  # [left, bottom, width, height]
    slider_a = Slider(ax_a, 'd core [um]', 0.35, w_CORE*2, valinit=w_CORE)
    slider_ML = []
    for ii, w_c in enumerate(w_list_CLAD):
        ax_b = plt.axes([0.18, 0.12-ii*0.03, 0.25, 0.03])
        slider_ML.append(Slider(ax_b, 'd layer '+str(ii+1) +
                         '[um]', 0.02, 3*w_c, valinit=w_c))


    # [left, bottom, width, height]
    # ax_a_n = plt.axes([0.68, 0.15, 0.25, 0.03])

    # slider_a_n = Slider(ax_a_n, 'n core [um]', 1, 3, valinit=n_CORE)
    # slider_ML_n = []
    # for ii, n_c in enumerate(n_list_CLAD):
    #     ax_b = plt.axes([0.68, 0.12-ii*0.03, 0.25, 0.03])
    #     slider_ML_n.append(
    #         Slider(ax_b, 'n layer '+str(ii+1) + '[um]', 1, 2, valinit=n_c))

    # Define the reset button
    resetax = plt.axes([0.68+0.25-0.1, 0.1, 0.1, 0.03])
    global button_reset #Needs to be initialized globally to remain responsive
    button_reset = Button(resetax, 'Reset', hovercolor='0.975')

    plt.subplots_adjust(top=0.96,
                        bottom=0.3,
                        left=0.1,
                        right=0.965,
                        hspace=0.345,
                        wspace=0.255)

    # Reset sliders to initial values
    def reset(event):
        slider_a.reset()
        # slider_a_n.reset()
        for ii in range(len(slider_ML)):
            slider_ML[ii].reset()
            # slider_ML_n[ii].reset()

        
    # Update function when sliders are changed
    def update(val):
        a = slider_a.val
        # a_n = slider_a_n.val
        b = []
        # b_n = []
        for ii in range(len(slider_ML)):
            b.append(slider_ML[ii].val)
            # b_n.append(slider_ML_n[ii].val)
        ### calculate the updated values
        n_eff_new = calc_n_eff_general(lambda_vector, n_CORE, n_SUB, n_list_CLAD, a, m, np.array(b))
        n_eff_smp_new = calc_n_eff_general(lambda_vector, n_CORE, n_SUB, n_list_CLAD[:,0], a, m)

        n_eff_smp_new_list = [item for sublist in n_eff_smp_new for item in sublist]
        n_eff_arr_new = reshape_n_eff(lambda_vector,n_eff_new)
        lasso_var.update_data(n_eff_arr_new)
        vpsn,  vgsn, betta2sn  = dispersion_calc(lambda_vector,n_eff_smp_new_list)
        
        line2.set_ydata(n_eff_smp_new)

        ax[1,0].cla()
        ax[1,0].plot(lambda_vector,  vgsn)
        ax[1,0].set_xlim(lambda_vector[0], lambda_vector[-1])

        ax[1,1].cla()
        ax[1,1].plot(lambda_vector[5:-2],  betta2sn[5:-2])
        ax[1,1].set_xlim(lambda_vector[0], lambda_vector[-1])
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def mouse_event(event):
        print('x: {} and y: {}'.format(event.xdata, event.ydata))

    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

    #Update and reset
    slider_a.on_changed(update)
    # slider_a_n.on_changed(update)
    for ii in range(len(slider_ML)):
        slider_ML[ii].on_changed(update)
        # slider_ML_n[ii].on_changed(update)

    button_reset.on_clicked(reset)
    plt.show()
    
    return 0



















def run_gui(lambda_vector, n_CORE, n_SUB, n_list_CLAD, w_CORE, w_list_CLAD, m):

    n_eff = np.array(calc_n_eff_ML(lambda_vector, n_CORE,
                     n_SUB, n_list_CLAD, w_CORE, w_list_CLAD, m))
    n_eff_simple = np.array(calc_n_eff(
        lambda_vector, n_CORE, n_SUB, n_list_CLAD[:,0], w_CORE, m))
    ### calculate vg, betta2
    vp,  vg, betta2  = dispersion_calc(lambda_vector,n_eff)
    vps,  vgs, betta2s  = dispersion_calc(lambda_vector,n_eff_simple)
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(2, 2)#, layout='constrained')
    # ax.set_title('Wavelength ')
    ###############   AX [0,0]   ###############
    l = ['MOLINO', 'Simple', 'Core index']
    line1, = ax[0,0].plot(lambda_vector,  n_eff, label=l[0])
    line2, = ax[0,0].plot(lambda_vector,  n_eff_simple, label=l[1])
    line3, = ax[0,0].plot(lambda_vector,  n_CORE, label=l[2], color='black', linestyle='--')
    ax[0,0].set_ylabel('Refractive index')
    ax[0,0].set_xlabel('Wavelength [$\mu$m]')
    ax[0,0].set_xlim(lambda_vector[0], lambda_vector[-1])
    ax[0,0].legend()
    
    ###############   AX [0,1]   ###############
    line_diff, = ax[0,1].plot(lambda_vector,  n_eff-n_eff_simple, 'k',label='Difference')
    ax[0,1].set_ylabel('Difference')
    ax[0,1].set_xlabel('Wavelength [$\mu$m]')
    ax[0,1].set_xlim(lambda_vector[0], lambda_vector[-1])
    formatter = ScalarFormatter(useOffset=True) # This will use scientific notation
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1)) 
    ax[0,1].yaxis.set_major_formatter(formatter)

    ###############   AX [1,0]   ###############
    
    line_vg1, = ax[1,0].plot(lambda_vector,  vg)
    line_vg2, = ax[1,0].plot(lambda_vector,  vgs)
    ax[1,0].set_ylabel('Group velocity')
    ax[1,0].set_xlabel('Wavelength [$\mu$m]')
    ax[1,0].set_xlim(lambda_vector[0], lambda_vector[-1])
    
    ###############   AX [1,1]   ###############
    
    line_disp1, = ax[1,1].plot(lambda_vector,  betta2)
    line_disp2, = ax[1,1].plot(lambda_vector,  betta2s)
    ax[1,1].set_ylabel('Group velocity dispersion')
    ax[1,1].set_xlabel('Wavelength [$\mu$m]')
    ax[1,1].set_xlim(lambda_vector[0], lambda_vector[-1])
    ax[1,1].set_ylim(-500,2000)
    
    ############### Define the sliders ###############
    ax_a = plt.axes([0.18, 0.15, 0.25, 0.03])  # [left, bottom, width, height]
    slider_a = Slider(ax_a, 'd core [um]', 0.35, w_CORE*2, valinit=w_CORE)
    slider_ML = []
    for ii, w_c in enumerate(w_list_CLAD):
        ax_b = plt.axes([0.18, 0.12-ii*0.03, 0.25, 0.03])
        slider_ML.append(Slider(ax_b, 'd layer '+str(ii+1) +
                         '[um]', 0.02, 10*w_c, valinit=w_c))


    # [left, bottom, width, height]
    # ax_a_n = plt.axes([0.68, 0.15, 0.25, 0.03])

    # slider_a_n = Slider(ax_a_n, 'n core [um]', 1, 3, valinit=n_CORE)
    # slider_ML_n = []
    # for ii, n_c in enumerate(n_list_CLAD):
    #     ax_b = plt.axes([0.68, 0.12-ii*0.03, 0.25, 0.03])
    #     slider_ML_n.append(
    #         Slider(ax_b, 'n layer '+str(ii+1) + '[um]', 1, 2, valinit=n_c))

    # Define the reset button
    resetax = plt.axes([0.68+0.25-0.1, 0.2, 0.1, 0.03])
    global button_reset #Needs to be initialized globally to remain responsive
    button_reset = Button(resetax, 'Reset', hovercolor='0.975')

    plt.subplots_adjust(top=0.96,
                        bottom=0.3,
                        left=0.1,
                        right=0.965,
                        hspace=0.345,
                        wspace=0.255)

    # Reset sliders to initial values
    def reset(event):
        slider_a.reset()
        # slider_a_n.reset()
        for ii in range(len(slider_ML)):
            slider_ML[ii].reset()
            # slider_ML_n[ii].reset()

        
    # Update function when sliders are changed
    def update(val):
        a = slider_a.val
        # a_n = slider_a_n.val
        b = []
        # b_n = []
        for ii in range(len(slider_ML)):
            b.append(slider_ML[ii].val)
            # b_n.append(slider_ML_n[ii].val)
        ### calculate the updated values
        n_eff_new = calc_n_eff_ML(lambda_vector, n_CORE, n_SUB, n_list_CLAD, a, b, m)
        n_eff_smp_new = calc_n_eff(lambda_vector, n_CORE, n_SUB, n_list_CLAD[:,0], a, m)
        n_diff = n_eff_new-n_eff_smp_new
        vpn,  vgn, betta2n  = dispersion_calc(lambda_vector,n_eff_new)
        vpsn,  vgsn, betta2sn  = dispersion_calc(lambda_vector,n_eff_smp_new)
        
        line1.set_ydata(n_eff_new)
        line2.set_ydata(n_eff_smp_new)
        line_diff.set_ydata(n_diff)
        line_vg1.set_ydata(vgn)
        line_vg2.set_ydata(vgsn)
        line_disp1.set_ydata(betta2n)
        line_disp2.set_ydata(betta2sn)

        ax[0,1].set_ylim(min(n_diff),max(n_diff)*1.1)
        fig.canvas.draw_idle()

    def mouse_event(event):
        print('x: {} and y: {}'.format(event.xdata, event.ydata))

    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

    #Update and reset
    slider_a.on_changed(update)
    # slider_a_n.on_changed(update)
    for ii in range(len(slider_ML)):
        slider_ML[ii].on_changed(update)
        # slider_ML_n[ii].on_changed(update)

    button_reset.on_clicked(reset)
    plt.show()
    
    return 0




def run_gui_simple(lambda_vector, n_CORE, n_SUB, n_list_CLAD, w_CORE, w_list_CLAD, m):

    n_eff = np.array(calc_n_eff_ML(lambda_vector, n_CORE,
                     n_SUB, n_list_CLAD, w_CORE, w_list_CLAD, m))
    n_eff_simple = np.array(calc_n_eff(
        lambda_vector, n_CORE, n_SUB, n_list_CLAD[0], w_CORE, m))
    fig, ax = plt.subplots()
    # ax.set_title('Wavelength ')
    l = ['MOLINO', 'Simple']
    line1, = ax.plot(lambda_vector,  n_eff, label=l[0])
    line2, = ax.plot(lambda_vector,  n_eff_simple, label=l[1])
    line3 = plt.axhline(y=n_CORE, color='black', linestyle='--')
    text1 = plt.text(lambda_vector[0], n_CORE, 'Core index', rotation=0,
                     verticalalignment='bottom', horizontalalignment='left')
    plt.ylabel('Refractive index')
    plt.xlabel('Wavelength [$\mu$m]')
    plt.ylim(1.5, 2.2)
    plt.xlim(lambda_vector[0], lambda_vector[-1])
    plt.legend()
    # plt.savefig('fig1.png',dpi = 500)

    # Define the sliders
    ax_a = plt.axes([0.18, 0.15, 0.25, 0.03])  # [left, bottom, width, height]
    slider_a = Slider(ax_a, 'd core [um]', 0.35, 1.5, valinit=w_CORE)
    slider_ML = []
    for ii, w_c in enumerate(w_list_CLAD):
        ax_b = plt.axes([0.18, 0.12-ii*0.03, 0.25, 0.03])
        slider_ML.append(Slider(ax_b, 'd layer '+str(ii+1) +
                         '[um]', 0.02, 1, valinit=w_c))

 
    ax_a_n = plt.axes([0.68, 0.15, 0.25, 0.03])
    slider_a_n = Slider(ax_a_n, 'n core [um]', 1, n_CORE*2, valinit=n_CORE)
    slider_ML_n = []
    for ii, n_c in enumerate(n_list_CLAD):
        ax_b = plt.axes([0.68, 0.12-ii*0.03, 0.25, 0.03])
        slider_ML_n.append(
            Slider(ax_b, 'n layer '+str(ii+1) + '[um]', 1, n_c*2, valinit=n_c))

    # Define the reset button
    resetax = plt.axes([0.68+0.25-0.1, 0.2, 0.1, 0.03])
    global button_reset #Needs to be initialized globally to remain responsive
    button_reset = Button(resetax, 'Reset', hovercolor='0.975')

    plt.subplots_adjust(top=0.96,
                        bottom=0.3,
                        left=0.125,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)

    # Reset sliders to initial values
    def reset(event):
        slider_a.reset()
        slider_a_n.reset()
        for ii in range(len(slider_ML)):
            slider_ML[ii].reset()
            slider_ML_n[ii].reset()

        
    # Update function when sliders are changed
    def update(val):
        ###  core values
        a = slider_a.val
        a_n = slider_a_n.val
        ###  cladding values
        b = []
        b_n = []
        for ii in range(len(slider_ML)):
            b.append(slider_ML[ii].val)
            b_n.append(slider_ML_n[ii].val)
        line1.set_ydata(calc_n_eff_ML(lambda_vector, a_n, n_SUB, b_n, a, b, m))
        line2.set_ydata(calc_n_eff(lambda_vector, a_n, n_SUB, b_n[0],a, m))
        line3.set_ydata(a_n)
        text1.set_position([lambda_vector[0], a_n])
        ax.set_ylim(1.5, a_n+0.1)
        # fig.canvas.draw_idle()

    def mouse_event(event):
        print('x: {} and y: {}'.format(event.xdata, event.ydata))

    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

    #Update and reset
    slider_a.on_changed(update)
    slider_a_n.on_changed(update)
    for ii in range(len(slider_ML)):
        slider_ML[ii].on_changed(update)
        slider_ML_n[ii].on_changed(update)

    button_reset.on_clicked(reset)

    # ax.on_clicked(add_dot)
    # button_calculate_n.on_clicked()
    plt.show()
    
    return 0
