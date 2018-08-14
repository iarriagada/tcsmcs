#!/usr/bin/env python


###########################################################
#
#	Analyze clock synchronization
#

from collections import namedtuple
from datetime import datetime, timedelta
from dateutil import tz
import argparse
import sys
import select
from multiprocessing import Process
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import dates
import matplotlib
import numpy as np
import re
import pytz
import pdb
import os
import time
# from pprint import pprint

sys.path.insert(0,'../')
sys.path.append('..')

from swglib.export import DataManager, get_exporter

plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.home'] = ''
plt.rcParams['keymap.grid'] = ''

TZ = 'America/Santiago'
PLOT_ZONE_FILE = './zonesV6.cfg'
LEGEND_LOCATION = 'lower left'
PLOT_AREA = (10,2)
BOTTOM_PLOT = True

TCS_SYSTEM = 'tcs'
TCS_CA = 'tcs:drives:driveMCS.VALA'
TCS_CASWG = 'tcs:drives:driveMCS.VALI'
# TCS_CA = 'tcs:drives:driveMCS.VALI'
MCS_SYSTEM = 'mcs'
MCSERR_CA = 'mc:AXISPmacPosError'
MCSDMD_CA = 'mc:AXISPmacDemandPos'
MCSCUR_CA = 'mc:AXISCurrentPos'
GWS_SYSTEM = 'gws'
GWS_CA = 'ws:tpUaTeSpeed3D'
WFS_SYSTEM = 'wfs'
WFS_CA = 'WFS:dc:ttf.VALTIPTILT'

FollowAArray = namedtuple('FollowAArray',\
                         'timestamp\
                         now\
                         targetTime\
                         trackId\
                         azPos\
                         elPos')
FollowIArray = namedtuple(\
                         'FollowIArray',\
                         'timestamp\
                         now\
                         traw\
                         tick\
                         applyDT\
                         dtraw\
                         dtick\
                         dtGetTelRD\
                         dmdCnt\
                         corrCnt\
                         dmdLowCorr\
                         dmdHighCorr\
                         fltCnt\
                         available\
                         azDmd\
                         elDmd'\
                         )
MCSArray = namedtuple('MCSArray', 'timestamp mcsVal')
GWSArray = namedtuple('GWSArray', 'timestamp windTopEnd')
WFSArray = namedtuple('WFSArray', 'timestamp wfsTipTilt')
ZoneArray = namedtuple('ZoneArray', 'title begin end color')
Limits = namedtuple('Limits', 'lower upper')

# Site should be either 'cp' or 'mk'
SITE = 'cp'
if SITE == 'cp':
    # directory where the data is located
    ROOT_DATA_DIR = '/archive/tcsmcs/data'
    if not os.path.exists(ROOT_DATA_DIR):
        ROOT_DATA_DIR =\
            '/net/cpostonfs-nv1/tier2/gem/sto/archiveRTEMS/tcsmcs/data'
else:
    raise NotImplementedError("The script hasn't been adapted for MK yet")

def parse_args():
    enableDebugDate = "debug"

    parser = argparse.ArgumentParser(
        description='Plot relevant data to TCS-MCS loop behaviour')

    parser.add_argument('detail',
                        metavar='DETAIL-LEVEL',
                        default='low',
                        help='Level of detail: \
                        \'low\'(tcs+mcs),\
                        \'medw\' (tcs+mcs+wfs),\
                        \'medp\' (tcs+mcs:pmac error/dmd),\
                        \'high\' (tcs+mcs+wfs+wind),\
                        \'swg\'(high + timing info)')

    parser.add_argument('date',
                        metavar='DATE',
                        help='Date to be analized in format YYYY-MM-DD')

    parser.add_argument('-tw',
                        '--timewindow',
                        dest='timeWindow',
                        nargs=2,
                        default=['18:00:00', '10:00:00'],
                        metavar=('INIT-TIME','END-TIME'),
                        help='Time window for the plot\
                        e.g.: -tw 18:00:00 06:00:00')

    parser.add_argument('-day',
                        '--day_mode',
                        dest='day_mode',
                        action='store_true',
                        help='If used daytime\
                        will be analyzed 6am-6pm')

    parser.add_argument('-z',
                        '--zoom',
                        dest='zoom',
                        action='store_true',
                        help='If used, plots will be zoomed out in Y-axis')

    parser.add_argument('-ax',
                        '--axis',
                        dest='axis',
                        default='all',
                        help='Axis to be analyzed (az, el, all) e.g.: -ax el')

    parser.add_argument('-wfs',
                        '--wfs',
                        dest='wfs',
                        default='p2',
                        help='Wavefront Sensor (p1, p2, oi)\
                        e.g.: -wfs p1')

    parser.add_argument('-tt',
                        '--tiptilt',
                        dest='tt',
                        default='all',
                        help='WFS Motion (tip, tilt, all) to be analyzed,\
                        e.g.: -tt tilt (Option used with \
                        \'swg\' level of detail only)')

    parser.add_argument('-cm',
                        '--ca_monitor',
                        dest='ca_mon',
                        action='store_true',
                        help='use this option if you are analyzing\
                        data captured with camonitor')

    parser.add_argument('-c',
                        '--cached_data',
                        dest='cached_data',
                        action='store_true',
                        help='use this option to capture data on the fly')

    args = parser.parse_args()

    return args

def dataPath(date, day_mode, system, PV):
    '''
    This generates the path for the specified system data
    '''
    day_str = "day_" if day_mode else ""
    #Get directory name
    pvdir = PV[PV.find(':')+1:].replace(':','_')
    #Construct the path to the data
    data_path = os.path.join(
        ROOT_DATA_DIR,
        SITE,
        system,
        pvdir,
        'txt',
        '{0}_{1}_{2}_{3}export.txt'.format(date,
                                           SITE,
                                           PV.replace(':','-'),
                                           day_str)
    )
    return data_path

def fromtimestampTz(ts):
    timezone = pytz.timezone(TZ)
    return datetime.fromtimestamp(ts, timezone)

def strptimeTz(dateStr, US=False):
    timezone = pytz.timezone(TZ)
    if US:
        when = datetime.strptime(dateStr, '%m/%d/%Y %H:%M:%S.%f')
    else:
        when = datetime.strptime(dateStr, '%Y-%m-%d %H:%M:%S')
    return timezone.localize(when)

def localizeNp64Tz(np64dt):
    """
    Converts from numpy.datetime64 to standard datetime with TZ
    """
    ts = ((np64dt - np.datetime64('1970-01-01T00:00:00Z'))
          / np.timedelta64(1, 's'))
    return fromtimestampTz(ts)

def producerT(pv, stime, etime):
    begin = strptimeTz(stime)
    end = strptimeTz(etime)
    tot_time = end - begin
    print 'Times:', begin, end
    dm = DataManager(get_exporter(SITE.upper()), root_dir='/tmp/rcm/')
    data = dm.getData(pv, begin, end)
    print "Processing", pv
    print "Starting"
    for val in data:
        if len(val) < 16:
            continue
        recTime = localizeNp64Tz(val[0])
        receptionTime = datetime.strftime(recTime, '%m/%d/%Y %H:%M:%S.%f')
        curr_time = datetime.strftime(recTime, '%Y-%m-%d %H:%M:%S')
        curr_time = strptimeTz(curr_time)
        receptionTime = datetime.strptime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        span_time = curr_time - begin
        progress = round((span_time.total_seconds()
                          / tot_time.total_seconds()) * 100, 2)
        sys.stdout.write('\r' + 'Progress: ' + str(progress) + '% ')
        # sys.stdout.write('\r' + 'Progress: ' + str(val) + '% ')
        sys.stdout.flush()
        # print "Spanned time:", span_time.total_seconds()
        yield FollowIArray(receptionTime, val[1], val[2], val[3], val[4],
                          val[5], val[6], val[7], val[8], val[9], val[10],
                          val[11], val[12], val[13], val[14], val[15])
    sys.stdout.write('\r' + 'Progress: 100.00%')
    sys.stdout.flush()
    print "\nDONE!"

def producerM(pv, stime, etime):
    # begin, end = getTimes(dateStr, timeRangeStr)
    begin = strptimeTz(stime)
    end = strptimeTz(etime)
    tot_time = end - begin
    # begin = stime
    # end = etime
    print 'Times:', begin, end
    dm = DataManager(get_exporter(SITE.upper()), root_dir='/tmp/rcm/')
    data = dm.getData(pv, begin, end)
    print "Processing", pv
    print "Starting"
    for val in data:
        if len(val) < 2:
            continue
        # receptionTime = localizeNp64Tz(val[0])
        # receptionTime = datetime.strftime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        # receptionTime = datetime.strptime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        recTime = localizeNp64Tz(val[0])
        receptionTime = datetime.strftime(recTime, '%m/%d/%Y %H:%M:%S.%f')
        curr_time = datetime.strftime(recTime, '%Y-%m-%d %H:%M:%S')
        curr_time = strptimeTz(curr_time)
        receptionTime = datetime.strptime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        span_time = curr_time - begin
        progress = round((span_time.total_seconds()
                          / tot_time.total_seconds()) * 100, 2)
        sys.stdout.write('\r' + 'Progress: ' + str(progress) + '% ')
        sys.stdout.flush()
        yield MCSArray(receptionTime, val[1])
    sys.stdout.write('\r' + 'Progress: 100.00%')
    sys.stdout.flush()
    print "\nDONE!"

def producerG(pv, stime, etime):
    # begin, end = getTimes(dateStr, timeRangeStr)
    begin = strptimeTz(stime)
    end = strptimeTz(etime)
    tot_time = end - begin
    # begin = stime
    # end = etime
    print 'Times:', begin, end
    dm = DataManager(get_exporter(SITE.upper()), root_dir='/tmp/rcm/')
    data = dm.getData(pv, begin, end)
    print "Processing", pv
    print "Starting"
    for val in data:
        if len(val) < 2:
            continue
        # receptionTime = localizeNp64Tz(val[0])
        # receptionTime = datetime.strftime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        # receptionTime = datetime.strptime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        recTime = localizeNp64Tz(val[0])
        receptionTime = datetime.strftime(recTime, '%m/%d/%Y %H:%M:%S.%f')
        curr_time = datetime.strftime(recTime, '%Y-%m-%d %H:%M:%S')
        curr_time = strptimeTz(curr_time)
        receptionTime = datetime.strptime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        span_time = curr_time - begin
        progress = round((span_time.total_seconds()
                          / tot_time.total_seconds()) * 100, 2)
        sys.stdout.write('\r' + 'Progress: ' + str(progress) + '% ')
        sys.stdout.flush()
        yield GWSArray(receptionTime, val[1])
    sys.stdout.write('\r' + 'Progress: 100.00%')
    sys.stdout.flush()
    print "\nDONE!"

def producerW(pv, stime, etime):
    # begin, end = getTimes(dateStr, timeRangeStr)
    begin = strptimeTz(stime)
    end = strptimeTz(etime)
    tot_time = end - begin
    # begin = stime
    # end = etime
    print 'Times:', begin, end
    dm = DataManager(get_exporter(SITE.upper()), root_dir='/tmp/rcm/')
    data = dm.getData(pv, begin, end)
    print "Processing", pv
    print "Starting"
    for val in data:
        if len(val) < 2:
            continue
        # receptionTime = localizeNp64Tz(val[0])
        # receptionTime = datetime.strftime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        # receptionTime = datetime.strptime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        recTime = localizeNp64Tz(val[0])
        receptionTime = datetime.strftime(recTime, '%m/%d/%Y %H:%M:%S.%f')
        curr_time = datetime.strftime(recTime, '%Y-%m-%d %H:%M:%S')
        curr_time = strptimeTz(curr_time)
        receptionTime = datetime.strptime(receptionTime, '%m/%d/%Y %H:%M:%S.%f')
        span_time = curr_time - begin
        progress = round((span_time.total_seconds()
                          / tot_time.total_seconds()) * 100, 2)
        sys.stdout.write('\r' + 'Progress: ' + str(progress) + '% ')
        sys.stdout.flush()
        yield WFSArray(receptionTime, val[1])
    sys.stdout.write('\r' + 'Progress: 100.00%')
    sys.stdout.flush()
    print "\nDONE!"

def producerTCS(filename):
    """
    This generates an array with the information harvested from the TCS
    """
    with open(filename) as source:
        # This indexes can be changed depending on the information origin i.e.
        # GEA or ca monitor
        rwIdx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        dl = '\t'
        # Change indexes if using ca monitor
        if args.ca_mon:
            rwIdx = rwIdx + 3
            dl = ' '
        reader = csv.reader(source, delimiter=dl)
        # Skip the headers
        next(reader); next(reader); next(reader); next(reader)
        # Populate TCS array
        for row in reader:
            if row[0].startswith('#') or row[1].startswith('#'):
                continue
            if args.ca_mon:
                if re.search('[a-z]', row[1]) or re.search('[a-z]', row[4]):
                    continue
                receptionTime = datetime.strptime(row[1]+' '+row[2],\
                                                  '%Y-%m-%d %H:%M:%S.%f')
            else:
                if float(row[1])==0:
                    continue
                receptionTime = datetime.strptime(row[0][:-3],\
                                                  '%m/%d/%Y %H:%M:%S.%f')
            yield FollowIArray(\
                              receptionTime, float(row[rwIdx[0]]),\
                              float(row[rwIdx[1]]), float(row[rwIdx[2]]),\
                              float(row[rwIdx[3]]), float(row[rwIdx[4]]),\
                              float(row[rwIdx[5]]), float(row[rwIdx[6]]),\
                              float(row[rwIdx[7]]), float(row[rwIdx[8]]),\
                              float(row[rwIdx[9]]), float(row[rwIdx[10]]),\
                              float(row[rwIdx[11]]), float(row[rwIdx[12]]),\
                              float(row[rwIdx[13]]), float(row[rwIdx[14]])\
                              )

def producerMCS(filename):
    """
    This generates an array with the information harvested from the MCS
    """
    with open(filename) as source:
        reader = csv.reader(source, delimiter='\t')
        # Skip the headers
        next(reader); next(reader); next(reader); next(reader)

        for row in reader:
            if row[0].startswith('#') or row[1].startswith('#'):
                continue
            yield MCSArray(\
                           datetime.strptime(row[0][:-3],\
                                             '%m/%d/%Y %H:%M:%S.%f'),\
                           float(row[1])\
                           )

def producerGWS(filename):
    """
    This generates an array with the information harvested from the GWS
    """
    with open(filename) as source:
        reader = csv.reader(source, delimiter='\t')
        # Skip the headers
        next(reader); next(reader); next(reader); next(reader)

        for row in reader:
            if row[0].startswith('#') or row[1].startswith('#'):
                continue
            yield GWSArray(\
                           datetime.strptime(row[0][:-3],\
                                             '%m/%d/%Y %H:%M:%S.%f'),\
                           float(row[1])\
                           )

def producerWFS(filename):
    """
    This generates an array with the information harvested from the WFS
    """
    with open(filename) as source:
        reader = csv.reader(source, delimiter='\t')
        # Skip the headers
        next(reader); next(reader); next(reader); next(reader)

        for row in reader:
            if row[0].startswith('#') or row[1].startswith('#'):
                continue
            yield WFSArray(\
                           datetime.strptime(row[0][:-3],\
                                             '%m/%d/%Y %H:%M:%S.%f'),\
                           float(row[1])\
                           )

def insideRange(begin, end, zone):
    """
    If part of the zone is inside the range adjust limits
    if its not inside the plotted range None is returned
    """
    if begin > zone.end or end < zone.begin:
        return None

    retZone = ZoneArray(zone.title, zone.begin, zone.end, zone.color)
    if zone.begin < begin:
        retZone = retZone._replace(begin=begin)
    if zone.end > end:
        retZone = retZone._replace(end=end)

    return retZone

def switchDateFormat(date, old_format, new_format):
    '''
    Switch input string "date" format style
    Return a tuple with new date as datetime object and as a string
    '''
    auxDate = datetime.strptime(date, old_format)
    newDateStr = datetime.strftime(auxDate, new_format)
    newDate = datetime.strptime(newDateStr, new_format)
    # Return a tuple with date as datetime object and as a string
    return [newDate, newDateStr]

def readZonesFromFile(begin, end):
    '''
    Read the information contained on the zones file.
    '''
    markedZones = []
    try:
        with open(PLOT_ZONE_FILE) as source:
            reader = csv.reader(source, skipinitialspace=True)
            for row in reader:
                if len(row)>3:
                    addFloat = ''
                    if not(row[1].find('.')+1):
                        addFloat = '.0'
                    initDate = switchDateFormat(row[1].strip() + addFloat,
                                                '%Y-%m-%d %H:%M:%S.%f',
                                                '%m/%d/%Y %H:%M:%S.%f')
                    addFloat = ''
                    if not(row[2].find('.')+1):
                        addFloat = '.0'
                    endDate = switchDateFormat(row[2].strip() + addFloat,
                                                '%Y-%m-%d %H:%M:%S.%f',
                                                '%m/%d/%Y %H:%M:%S.%f')
                    za = ZoneArray(row[0].strip(),
                                   initDate[0],
                                   endDate[0],
                                   row[3].strip())
                    za = insideRange(begin, end, za)
                    if za:
                        markedZones.append(za)
    except Exception as ex:
        print "Exception reading zones file:"
        print ex
        pass
    # print markedZones[0].title
    return markedZones

def addZones(ax, begin, end):
    '''
    Plot the information defined on the zones files. Only the information specific
    for the day specified on the arguments will be plotted.
    '''
    markedZones = readZonesFromFile(begin, end)
    zones = []

    for ii, za in enumerate(markedZones):
        zz = ax.axvspan(za.begin, za.end, facecolor=za.color,
                   label=za.title, alpha=0.3)
        zones.append(zz)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    # ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize = 'small')
    return zones


def plotConfig(ax_lst):
    '''
    This function is used to:
        - Disable tick labels for plots other than the bottom positions
        - Set the date labels on the bottom plots
        - Update Plot Event Zones
        '''
    for ax in ax_lst:
        ax[0].ax.grid(True)
        if not(ax[1]):
            plt.setp(ax[0].ax.get_xticklabels(), fontsize=9, visible=False)
        else:
            ax[0].ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
            ax[0].ax.xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter("%d/%m %H:%M:%S.%f"))
            ax[0].ax.xaxis.set_minor_locator(ticker.MaxNLocator(200))
            plt.setp(ax[0].ax.get_xticklabels(), fontsize=9,
                     rotation=30, ha='right')
            ax[0].ax.set_xlim(lowX,highX)
            # h, l = ax[0].ax.get_legend_handles_labels()
            # ax[0].ax.legend(h, l, loc='upper right', bbox_to_anchor=(1, 1), fontsize = 'small')
            ax[0].ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize = 'small')

class AxPlt:
    '''
    This class defines a plot object
    '''
    # Attributes common to every instance
    plotArea = None
    sharedAx = []
    shareFlag = []

    def __init__(self, linestyle, ylabel,
                 ylims=(0,0), autoyax=True):
        '''
        Creates the plot obejct
        '''
        self.linestyle = linestyle
        self.ylabel = ylabel
        self.autoyax = autoyax
        self.ylims = ylims
        self.ax = None
        self.dataT = None
        self.dataY = None

    def plot_ax(self, ax_lst, position, height, width,
                data, bottomPlot=False, errorLine=False):
        self.dataT, self.dataY = zip(*data)
        if not(self.shareFlag):
            self.ax = plt.subplot2grid(self.plotArea, position,
                                       colspan = width,
                                       rowspan = height)
            self.sharedAx.append(self.ax)
            self.shareFlag.append(True)
        else:
            self.ax = plt.subplot2grid(self.plotArea, position,
                                       colspan = width,
                                       rowspan = height,
                                       sharex=self.sharedAx[0])

        self.ax.plot(self.dataT, self.dataY, self.linestyle)
        if errorLine:
            self.ax.axhline(y=1.0, linestyle='-.', linewidth=1.25, color='crimson')
            self.ax.axhline(y=0.5, linestyle='-.', linewidth=1.25, color='darkorange')
            self.ax.axhline(y=-1.0, linestyle='-.', linewidth=1.25, color='crimson')
            self.ax.axhline(y=-0.5, linestyle='-.', linewidth=1.25, color='darkorange')
        self.ax.grid(True)
        self.ax.tick_params("y", colors="b")
        self.ax.set_ylabel(self.ylabel, color="b")
        if not(self.autoyax):
            self.ax.set_ylim(self.ylims[0], self.ylims[1])
        ax_lst.append([self, bottomPlot])
# axPlt end

def on_xlims_change(ax):
    global saveFlag
    if saveFlag:
        global zxlims
        global zxlimsDate
        ninit = dates.num2date(ax.get_xlim()[0])
        nend = dates.num2date(ax.get_xlim()[1])
        ninit = datetime.strftime(ninit, "%Y-%m-%d %H:%M:%S.%f")
        nend = datetime.strftime(nend, "%Y-%m-%d %H:%M:%S.%f")
        ninitDate = datetime.strptime(ninit, "%Y-%m-%d %H:%M:%S.%f")
        nendDate = datetime.strptime(nend, "%Y-%m-%d %H:%M:%S.%f")
        zxlimsDate = [ninitDate, nendDate]
        zxlims =  ninit + ", " + nend
        print zxlims

def on_ylims_change(ax):
    print "updated ylims: ", ax.get_ylim()

def ylim_shout(ax):
    print "Y Lim changed!!!!!!"

def save_zone(event):
    global zxlims
    global zxlimsDate
    global ax_lst
    global saveFlag
    global fig
    global evZones
    if (event.key == 's') and saveFlag:
        saveFlag = False
        print "you pressed ", event.key
        zname = raw_input("Enter zoom name (\'a\' to abort): ")
        if zname == 'a':
            print 'Abort'
            return
        zcol = raw_input("Choose color (r/y/g): ")
        write_zone(zname, zcol, zxlims)
        remove_zones(evZones)
        evZones = paint_zones(ax_lst)
        plotConfig(ax_lst)
        fig.canvas.draw()
        # i = 0
        # for auxax in ax_lst:
            # auxzone = auxax[0].ax.axvspan(zxlimsDate[0], zxlimsDate[1],
                                          # facecolor=zcol,
                                          # label=zname,
                                          # alpha=0.3)
            # # evZones.append([auxzone])
            # evZones[i].append(auxzone)
            # auxax[0].ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
                               # fontsize = 'small')
            # i += 1
        # fig.canvas.draw()
        saveFlag = True

def select_zone(event):
    global fig
    global evZones
    global ax_lst
    if (event.key == 'z'):
        currZones = disp_zones(ax_lst)
        zoomOpt = raw_input('Select area to zoom [\'a\' to abort]:')
        if zoomOpt == 'a':
            print 'Zoom Aborted'
            return
        ax_lst[0][0].ax.set_xlim(currZones[int(zoomOpt)-1].begin, currZones[int(zoomOpt)-1].end)
        fig.canvas.draw()

def zoom_out(event):
    global fig
    global ax_lst
    global lowX
    global highX
    if (event.key == 'h'):
        print 'Zooming out'
        ax_lst[0][0].ax.set_xlim(lowX, highX)
        fig.canvas.draw()

def enable_event(event):
    global evZones
    if event.key == 't':
        print "You pressed Enter!"
        ezlist = [izone for ezones in evZones for izone in ezones]
        for ez in ezlist:
            visState = ez.get_visible()
            ez.set_visible(not(visState))
        fig.canvas.draw()

def delete_zone(event):
    global fig
    global ax_lst
    global evZones
    if (event.key == 'd'):
        currZones = disp_zones(ax_lst)
        zoneDel = raw_input('Select area to delete [\'a\' to abort]:')
        if zoneDel == 'a':
            print 'Delete Aborted'
            return
        with open(PLOT_ZONE_FILE, 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            newlines = [line for line in lines if (line.find(currZones[int(zoneDel)-1].title) == -1)]
            f.writelines(newlines)
            f.truncate()
        remove_zones(evZones)
        evZones = paint_zones(ax_lst)
        plotConfig(ax_lst)
        fig.canvas.draw()

def hot_keys_ref(event):
    if (event.key == 'r'):
        hot_keys_disp()

def remove_zones(evZones):
    ezlist = [izone for ezones in evZones for izone in ezones]
    for ez in ezlist:
        visState = ez.remove()

def write_zone(name, col, xlims):
    newzone = name + ", " + xlims + ", " + col + "\n"
    # f = open(PLOT_ZONE_FILE, 'a')
    with open(PLOT_ZONE_FILE, 'a') as f:
        f.write(newzone)
        # f.close()
    print "Zone saved: ", newzone

def disp_zones(ax_lst):
    currZones = readZonesFromFile(ax_lst[0][0].dataT[0], ax_lst[0][0].dataT[-1])
    print '******************** List of Zones ********************'
    zoom_lst = []
    i = 1
    for dispZones in currZones:
        llDate = datetime.strftime(dispZones.begin, "%Y-%m-%d %H:%M:%S.%f")
        hlDate = datetime.strftime(dispZones.end, "%Y-%m-%d %H:%M:%S.%f")
        print '[{0}] '.format(i) + dispZones.title + ' ' + llDate + ' -> ' + hlDate
        i += 1
    print '*******************************************************'
    return currZones

def paint_zones(ax_lst):
    evZns = []
    for axplt in ax_lst:
        evZns.append(addZones(axplt[0].ax, axplt[0].dataT[0], axplt[0].dataT[-1]))
    return evZns

def hot_keys_disp():
    print '**********************************************'
    print '** Hot keys quick reference guide'
    print '** r: Display this quick reference'
    print '** s: Save current zoom as zone of interest'
    print '** d: Delete zone of interest'
    print '** z: Zoom to selected zone of interest'
    print '** h: Zoom out'
    print '**********************************************'

if __name__ == '__main__':
# def main():
    # ---------- MAIN ----------
    args = parse_args()

    # DATA ARRAYS GENERATION SECTION

    # Switches input argument "date" format to match date format from GEA. This
    # doesn't look very elegant, I know...
    dateP = switchDateFormat(args.date, '%Y-%m-%d', '%m/%d/%Y')

    # Set X-axis plot limits
    lowX = datetime.strptime(dateP[1] + ' ' + args.timeWindow[0] + '.0',
                            '%m/%d/%Y %H:%M:%S.%f')
    highX = datetime.strptime(dateP[1] + ' ' + args.timeWindow[1] + '.0',
                            '%m/%d/%Y %H:%M:%S.%f')
    startNight = datetime.strptime(dateP[1] + ' ' + '18:00:00.0',
                                '%m/%d/%Y %H:%M:%S.%f')
    midNight = datetime.strptime(dateP[1] + ' ' + '23:59:59.9',
                                '%m/%d/%Y %H:%M:%S.%f')

    # Checks to see if X-axis low limit falls after midnight and corrects date.
    if not(args.day_mode):
        if (lowX < startNight) and not(args.cached_data):
            lowX = lowX + timedelta(days=1)
        # Checks to see if X-axis high limit falls after midnight and corrects date.
        if highX < lowX:
            highX = highX + timedelta(days=1)
    else:
        auxX = lowX
        lowX = highX
        highX = auxX

    lowX_cache = datetime.strftime(lowX,'%Y-%m-%d %H:%M:%S')
    highX_cache = datetime.strftime(highX,'%Y-%m-%d %H:%M:%S')

    # Start of TCS data management
    tcs_data_path = dataPath(args.date, args.day_mode, TCS_SYSTEM, TCS_CA)
    print "Reading: {0}".format(tcs_data_path)
    # flw_producer = producerTCS(tcs_data_path)
    if args.detail in ['swg']:
        flw_producer = producerT(TCS_CASWG, lowX_cache, highX_cache)
    else:
        flw_producer = producerT(TCS_CASWG, lowX_cache, highX_cache)
        # flw_producer = producerT(TCS_CA, lowX_cache, highX_cache)

    # print 'Done with {0}'.format(TCS_CA)


    diff_lst = list()
    exec_lst = list()
    dDmdAz_lst = list()
    dDmdEl_lst = list()
    vDmdAz_lst = list()
    vDmdEl_lst = list()
    dmdAz_lst = list()
    dmdEl_lst = list()
    corr_lst = list()
    flt_lst = list()
    dmdC_lst = list()
    applyDt_lst = list()
    sendDt_lst = list()
    dtGTRD_lst = list()

    prevElDmd = flw_producer.next().elDmd
    prevAzDmd = flw_producer.next().azDmd
    prevCorrCnt = flw_producer.next().corrCnt
    prevDmdCnt = flw_producer.next().dmdCnt
    prevFltCnt = flw_producer.next().fltCnt

    outliersInPeriod = 0
    periodLimits = Limits(-1,1)



    for dp in flw_producer:
        dElDemands = dp.elDmd - prevElDmd
        dAzDemands = dp.azDmd - prevAzDmd
        dCorrCnt = dp.corrCnt - prevCorrCnt
        dDmdCnt = dp.dmdCnt - prevDmdCnt
        dFltCnt = dp.fltCnt - prevFltCnt

        # if dDmdCnt > 1 and ( dCorrCnt == 0  or dFltCnt == 0):
        if ( dDmdCnt > 1 ) and ( dCorrCnt == 0 ):
            dAzDemands = dAzDemands / 2
            dElDemands = dElDemands / 2

        diff_lst.append((dp.timestamp, dp.dtraw*1000.0))
        exec_lst.append((dp.timestamp, dp.dtick*1000.0))
        dDmdEl_lst.append((dp.timestamp, dElDemands*3600))
        dDmdAz_lst.append((dp.timestamp, dAzDemands*3600))

        if dp.dtraw > 0:
            vAzDemands = dAzDemands / dp.dtraw
            vElDemands = dElDemands / dp.dtraw
            vDmdAz_lst.append((dp.timestamp, vAzDemands*3600))
            vDmdEl_lst.append((dp.timestamp, vElDemands*3600))

        applyDt_lst.append((dp.timestamp, dp.applyDT*1000.0))
        # dtfl_lst.append((dp.timestamp, dp.fltTime*1000.0))
        corr_lst.append((dp.timestamp, dCorrCnt))
        flt_lst.append((dp.timestamp, dFltCnt))
        dtGTRD_lst.append((dp.timestamp, dp.dtGetTelRD))
        dmdC_lst.append((dp.timestamp, dDmdCnt))
        dmdEl_lst.append((dp.timestamp, dp.elDmd))
        dmdAz_lst.append((dp.timestamp, dp.azDmd))
        prevElDmd = dp.elDmd
        prevAzDmd = dp.azDmd
        prevCorrCnt = dp.corrCnt
        prevFltCnt = dp.fltCnt
        prevDmdCnt = dp.dmdCnt

        # if dp.dtraw > periodLimits.upper\
                # or\
                # dp.traw < periodLimits.lower:
            # print "Period out of limits: {0} on date: {1}".format(\
                                                                # dp.dtraw,\
                                                                # dp.timestamp\
                                                                # )
            # outliersInPeriod += 1

    print dp.timestamp

    # Start of MCS, GWS, WFS data management. This checks the input arguments
    # corresponding to each system
    if args.axis in ['az', 'all']:
        # Start of MCS data
        mcsAz_lst = list()
        PV = MCSERR_CA
        PV = PV.replace('AXIS','az')

        # mcsAz_data_path = dataPath(args.date, args.day_mode, MCS_SYSTEM, PV)
        # print "Reading: {0}".format(mcsAz_data_path)
        # mcs_producer = producerMCS(mcsAz_data_path)
        mcs_producer = producerM(PV, lowX_cache, highX_cache)
        for dp in mcs_producer:
            mcsAz_lst.append((dp.timestamp, dp.mcsVal*3600))

        mcsAz_lst = mcsAz_lst[:-1]
        if args.detail in ['medp']:
            mcsDmdAz_lst = list()
            PV = MCSDMD_CA
            PV = PV.replace('AXIS','az')

            # mcsAz_data_path = dataPath(args.date, args.day_mode, MCS_SYSTEM, PV)
            # print "Reading: {0}".format(mcsAz_data_path)
            # mcs_producer = producerMCS(mcsAz_data_path)
            mcs_producer = producerM(PV, lowX_cache, highX_cache)
            for dp in mcs_producer:
                mcsDmdAz_lst.append((dp.timestamp, dp.mcsVal))

    if ((args.axis in ['el', 'all']) and not(args.detail in ['swg']))\
    or ((args.axis in ['el']) and (args.detail in ['swg'])):
        # Start of MCS data
        mcsEl_lst = list()
        PV = MCSERR_CA
        PV = PV.replace('AXIS','el')

        # mcsEl_data_path = dataPath(args.date, args.day_mode, MCS_SYSTEM, PV)
        # print "Reading: {0}".format(mcsEl_data_path)
        # mcs_producer = producerMCS(mcsEl_data_path)
        mcs_producer = producerM(PV, lowX_cache, highX_cache)
        for dp in mcs_producer:
            mcsEl_lst.append((dp.timestamp, dp.mcsVal*3600))

        if args.detail in ['medp']:
            mcsDmdEl_lst = list()
            PV = MCSDMD_CA
            PV = PV.replace('AXIS','el')

            # mcsEl_data_path = dataPath(args.date, args.day_mode, MCS_SYSTEM, PV)
            # print "Reading: {0}".format(mcsEl_data_path)
            # mcs_producer = producerMCS(mcsEl_data_path)
            mcs_producer = producerM(PV, lowX_cache, highX_cache)
            for dp in mcs_producer:
                mcsDmdEl_lst.append((dp.timestamp, dp.mcsVal))

    if args.detail in ['high', 'swg']:
        # Start of GWS data
        gws_lst = list()
        # gws_data_path = dataPath(args.date, args.day_mode, GWS_SYSTEM, GWS_CA)
        # print "Reading: {0}".format(gws_data_path)
        # gws_producer = producerGWS(gws_data_path)
        gws_producer = producerG(GWS_CA, lowX_cache, highX_cache)
        for dp in gws_producer:
            gws_lst.append((dp.timestamp, dp.windTopEnd))

    if args.detail in ['medw', 'high']:
        # Start of WFS data
        wfsA_lst = list()
        wfsB_lst = list()
        PV = WFS_CA

        if args.wfs in ['p1']:
            PV = PV.replace('WFS','pwfs1')
        elif args.wfs in ['p2']:
            PV = PV.replace('WFS','pwfs2')
        elif args.wfs in ['oi']:
            PV = PV.replace('WFS','oiwfs')

        if args.tt in ['tip', 'all']:
            PV = PV.replace('TIPTILT','A')
            # wfsA_data_path = dataPath(args.date, args.day_mode, WFS_SYSTEM, PV)
            # print "Reading: {0}".format(wfsA_data_path)
            # wfs_producer = producerWFS(wfsA_data_path)
            wfs_producer = producerW(PV, lowX_cache, highX_cache)
            for dp in wfs_producer:
                wfsA_lst.append((dp.timestamp, dp.wfsTipTilt))
            PV = PV.replace('VALA','VALTIPTILT')

        if args.tt in ['tilt', 'all']:
            PV = PV.replace('TIPTILT','B')
            # wfsB_data_path = dataPath(args.date, args.day_mode, WFS_SYSTEM, PV)
            # print "Reading: {0}".format(wfsB_data_path)
            # wfs_producer = producerWFS(wfsB_data_path)
            wfs_producer = producerW(PV, lowX_cache, highX_cache)
            for dp in wfs_producer:
                wfsB_lst.append((dp.timestamp, dp.wfsTipTilt))


    # PLOTTING SECTION
    print "**********"
    print "Generating Plots..."
    print "**********"
    hot_keys_disp()

    fig = plt.figure()
    zxlims = '' # Variable initialization for Zones x-axis limits
    zxlimsDate = [] # Variable initialization for Zones x-axis limits
    saveFlag = True #Flag to enable/disable saving plot zooms
    AxPlt.plotArea = PLOT_AREA

    # Define Y-Label text for several plots
    # Define label for mount axis
    ElTitle = 'Elevation'
    AzTitle = 'Azimuth'

    # Define label for WFS
    if (args.wfs == 'p1'):
        wfsTitle = 'Pwfs1'
    elif (args.wfs == 'p2'):
        wfsTitle = 'Pwfs2'
    elif (args.wfs == 'oi'):
        wfsTitle = 'Oiwfs'

    wfsTitleA = wfsTitle + ' tip'
    wfsTitleB = wfsTitle + ' tilt'

    # Initialize plots
    # Set Y-axis zoom limits
    autoyax = True
    dmdYLim = (0,0)
    pmacYLim = (0,0)
    wfsYLim = (0,0)

    timeYLim = (-50, 150)
    cntYLim = (0, 4)
    appDtYLim = (-10, 200)

    if not(args.zoom):
        autoyax = False
        dmdYLim = (-10,10)
        pmacYLim = (-15,15)
        wfsYLim = (-0.5,0.5)

    # Initialize TCS Time plots
    tcsDTraw = AxPlt('r.-', "Diff traw\n[ms]", timeYLim, False)
    tcsDTick = AxPlt('r.-', "Diff tick\n[ms]", timeYLim, False)
    tcsAppTime = AxPlt('r.-', "Tick - Traw\n[ms]", timeYLim, False)
    tcsDCorrCnt = AxPlt('r.-', "Diff\nCorr Cnt\n[un]", cntYLim, False)
    tcsDDmdCnt = AxPlt('r.-', "Diff\nDmd Cnt\n[un]", cntYLim, False)
    tcsFltCnt = AxPlt('r.-', "Diff\nFault Cnt\n[un]", cntYLim, False)
    tcsGetTelRD = AxPlt('r.-', "Diff\nGetTelRaDec\nExecution\n[ms]")

    # Initialize Az plots
    tcsAzDmd = AxPlt('g-', "{0} Demand\n[degrees]".format(AzTitle))
    tcsAzdDmd = AxPlt('g-', "Diff\n{0} Demand\n[arcsec]".format(AzTitle), dmdYLim, autoyax)
    mcsAzPmErr = AxPlt('b-', "{0}\nPmac Error\n[arcsec]".format(AzTitle), pmacYLim, autoyax)
    mcsAzPmDmd = AxPlt('b-', "{0}\nPmac Demand\n[degrees]".format(AzTitle))

    # Initialize El plots
    tcsElDmd = AxPlt('g-', "{0} Demand\n[degress]".format(ElTitle))
    tcsEldDmd = AxPlt('g-', "Diff\n{0} Demand\n[arcsec]".format(ElTitle), dmdYLim, autoyax)
    mcsElPmErr = AxPlt('b-', "{0}\nPmac Error\n[arcsec]".format(ElTitle), pmacYLim, autoyax)
    mcsElPmDmd = AxPlt('b-', "{0}\nPmac Demand\n[degrees]".format(ElTitle))

    # Initialize WFS plots
    wfsTip = AxPlt('b-', "{0}\n[arcsec]".format(wfsTitleA), wfsYLim, autoyax)
    wfsTilt = AxPlt('b-', "{0}\n[arcsec]".format(wfsTitleB), wfsYLim, autoyax)

    # Initialize Wind plot
    gwsWind = AxPlt('g-', "Wind Speed\nTop End\n[m/s]")

    # Layout settings
    # In case the reader wants to generate their own plotting scheme, the setup
    # needs to follow this simple structure
    # NAME_OF_PLOT_OBJECT.plot.ax([position parameters]+[data list])
    # The order in wich the plots are called on the script is the plot number. The
    # function plot_ax puts its own plot object in a list, which can then be used
    # to iteratively access each plot object.

    ax_lst = list() # Initialize plot axis list

    # Detail level Low
    if args.detail in ['low']:
        AxPlt.plotArea = (4,2)
        if args.axis in ['az','all']:
            cs = 1
            if args.axis in ['az']:
                cs = 2
            mcsAzPmErr.plot_ax(ax_lst, (0,0), 1, cs, mcsAz_lst, errorLine=True)
            tcsAzdDmd.plot_ax(ax_lst, (1,0), 1, cs, dDmdAz_lst)
            tcsAzDmd.plot_ax(ax_lst, (2,0), 2, cs, dmdAz_lst, BOTTOM_PLOT)

        if args.axis in ['el','all']:
            cs = 1
            cp = 1
            if args.axis in ['el']:
                cs = 2
                cp = 0
            mcsElPmErr.plot_ax(ax_lst, (0,cp), 1, cs, mcsEl_lst, errorLine=True)
            tcsEldDmd.plot_ax(ax_lst, (1,cp), 1, cs, dDmdEl_lst)
            tcsElDmd.plot_ax(ax_lst, (2,cp), 2, cs, dmdEl_lst, BOTTOM_PLOT)

    # Detail level Med and High
    if args.detail in ['medw', 'medp', 'high']:
        AxPlt.plotArea = (6,2) # Define plotting area
        p0index = 0
        p1index = 0
        rs = 3
        p1step = 3
        bottomR = False
        bottomL = False
        if args.axis in ['az','all']:
            # Change conditions if only Az axis has been requested
            if args.axis in ['az']:
                AxPlt.plotArea = (4,2) # Define plotting area
                rs = 4
                if args.detail in ['medp']:
                    AxPlt.plotArea = (3,2) # Define plotting area
                    rs = 3
                    bottomL = True
                bottomR = True
            if args.detail in ['medp']:
                mcsAzPmDmd.plot_ax(ax_lst, (p0index,0), 1, 1, mcsDmdAz_lst)
                p0index += 1
            mcsAzPmErr.plot_ax(ax_lst, (p0index,0), 1, 1, mcsAz_lst, errorLine=True)
            p0index += 1
            tcsAzdDmd.plot_ax(ax_lst, (p0index,0), 1, 1, dDmdAz_lst, bottomL)
            p0index += 1
            # Add Wind information in case plot detail set to High
            if args.detail in ['high']:
                rs = 2
                p1step = 2
                gwsWind.plot_ax(ax_lst, (p1index,1), rs, 1, gws_lst)
                p1index += p1step
            tcsAzDmd.plot_ax(ax_lst, (p1index,1), rs, 1, dmdAz_lst, bottomR)
            p1index += p1step

        if args.axis in ['el','all']:
            # Change conditions if only El axis has been requested
            if args.axis in ['el']:
                AxPlt.plotArea = (4,2) # Define plotting area
                rs = 4
                if args.detail in ['medp']:
                    AxPlt.plotArea = (3,2) # Define plotting area
                    rs = 3
            if args.detail in ['medp']:
                bottomL = True
                mcsElPmDmd.plot_ax(ax_lst, (p0index,0), 1, 1, mcsDmdEl_lst)
                p0index += 1
            mcsElPmErr.plot_ax(ax_lst, (p0index,0), 1, 1, mcsEl_lst, errorLine=True)
            p0index += 1
            tcsEldDmd.plot_ax(ax_lst, (p0index,0), 1, 1, dDmdEl_lst, bottomL)
            p0index += 1
            # Add Wind information in case plot detail set to High
            # P1index is used to determine if Wind has already been used
            if (args.detail in ['high']) and (p1index == 0):
                rs = 2
                p1step = 2
                gwsWind.plot_ax(ax_lst, (p1index,1), rs, 1, gws_lst)
                p1index += p1step
            tcsElDmd.plot_ax(ax_lst, (p1index,1), rs, 1, dmdEl_lst, BOTTOM_PLOT)

        if args.detail in ['medw', 'high']:
            wfsTip.plot_ax(ax_lst, (p0index,0), 1, 1, wfsA_lst)
            p0index += 1
            wfsTilt.plot_ax(ax_lst, (p0index,0), 1, 1, wfsB_lst, BOTTOM_PLOT)

    # Detail Level SWG
    if args.detail in ['swg']:
        AxPlt.plotArea = (10,2) # Define plotting area
        tcsDTraw.plot_ax(ax_lst, (0,0), 2, 1, diff_lst)
        tcsDTick.plot_ax(ax_lst, (2,0), 2, 1, exec_lst)
        tcsAppTime.plot_ax(ax_lst, (4,0), 2, 1, applyDt_lst)
        tcsGetTelRD.plot_ax(ax_lst, (6,0), 1, 1, dtGTRD_lst)
        tcsDCorrCnt.plot_ax(ax_lst, (7,0), 1, 1, corr_lst)
        tcsDDmdCnt.plot_ax(ax_lst, (8,0), 1, 1, dmdC_lst)
        tcsFltCnt.plot_ax(ax_lst, (9,0), 1, 1, flt_lst, BOTTOM_PLOT)
        gwsWind.plot_ax(ax_lst, (0,1), 2, 1, gws_lst)
        if args.axis in ['az', 'all']:
            mcsAzPmErr.plot_ax(ax_lst, (2,1), 2, 1, mcsAz_lst, errorLine=True)
            tcsAzdDmd.plot_ax(ax_lst, (4,1), 2, 1, dDmdAz_lst)
            tcsAzDmd.plot_ax(ax_lst, (6,1), 4, 1, dmdAz_lst, BOTTOM_PLOT)
        else:
            mcsElPmErr.plot_ax(ax_lst, (2,1), 2, 1, mcsEl_lst, errorLine=True)
            tcsEldDmd.plot_ax(ax_lst, (4,1), 2, 1, dDmdEl_lst)
            tcsElDmd.plot_ax(ax_lst, (6,1), 4, 1, dmdEl_lst, BOTTOM_PLOT)



    print "Plotting window time limits"
    print lowX
    print highX
    print "Number of detected period outliers: {0}".format(outliersInPeriod)

    evZones = []
    evZones = paint_zones(ax_lst)

    # Configure axis format and labels for every plot
    plotConfig(ax_lst)

    plt.suptitle('TCS-MCS Analysis Plots\nDate: {0} - Detail Level: {1}'.format(
        args.date, args.detail))

    for auxax in ax_lst:
        auxax[0].ax.callbacks.connect('xlim_changed', on_xlims_change)

    # ax_lst[0][0].ax.callbacks.connect('xlim_changed', on_xlims_change)
    # ax_lst[0][0].ax.callbacks.connect('ylim_changed', on_ylims_change)
    # ax_lst[0][0].ax.callbacks.connect('ylim_changed', ylim_shout)

    fig.canvas.mpl_connect('key_press_event', save_zone)
    fig.canvas.mpl_connect('key_press_event', enable_event)
    fig.canvas.mpl_connect('key_press_event', select_zone)
    fig.canvas.mpl_connect('key_press_event', zoom_out)
    fig.canvas.mpl_connect('key_press_event', delete_zone)
    fig.canvas.mpl_connect('key_press_event', hot_keys_ref)

    # Display plot on separate window
    plt.show()
