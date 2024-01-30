from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage
import uuid

## To send hyperparameters to Unity environment
class HyperParametersSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d"))  
        self.received_rewards = []  # Store the received data here

    #def on_message_received(self, msg: IncomingMessage) -> None:
        #Number of hyperparameters
     #   count = msg.read_int32() 
      #  rewards = [msg.read_float32() for _ in range(count)]
       # self.received_rewards = rewards  # Store the received data
    def on_message_received(self, msg: IncomingMessage) -> None:
        pass
    def send_hyperparameters(self, float_list):
        msg = OutgoingMessage()
        msg.write_int32(len(float_list))  # Write the count of float values
        for value in float_list:
            msg.write_float32(value)  # Write each float value
        self.queue_message_to_send(msg)
