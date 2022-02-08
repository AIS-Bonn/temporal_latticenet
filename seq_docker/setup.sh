echo "Performing setup of the system by writing some config files"
#ros sourcing
./echo_to_file.sh ~/.bashrc "source /opt/ros/melodic/setup.bash"
#./echo_to_file.sh ~/.bashrc "source /media/rosu/Data/phd/c_ws/devel/setup.bash"

echo "Setting up Kdenlive by exporting dbus-launch"
# ./echo_to_file ~/.bashrc "export $(dbus-launch)"
#activate conda environment which is the same in which we install stuff in the Dockerfile
# ./echo_to_file.sh ~/.bashrc "source activate pt"
#make it so that conda doesnt modify the terminal line when we activate the conda environment
# touch ~/.condarc
# ./echo_to_file.sh ~/.condarc "changeps1: false"

#some nice hosts to have
# ./echo_to_file.sh /etc/hosts "10.7.3.52       mbzirc2"
# ./echo_to_file.sh /etc/hosts "10.7.3.57       mbzirc7"
# ./echo_to_file.sh /etc/hosts "131.220.7.55       bigcuda5"
# ./echo_to_file.sh /etc/hosts "10.7.3.180       drz1"

#for photoneo phoxi controler to work we need to start this 
#sudo /etc/init.d/dbus start
#sudo /etc/init.d/avahi-daemon start
