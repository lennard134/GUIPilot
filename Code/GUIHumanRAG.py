import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
import random
#Helper functions from the provided code


def document_parsing(file_path, chunk_size):
    """
    Document parser, processes uploaded document and splits text into chunks for a given chunksize.
    """
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_size/10, length_function=len)
    return text_splitter.split_documents(pages)

#GUI Implementation
class ChatApp:
    # TODO: Summarize chat history before feeding back into model.
    
    def __init__(self, root:tk.Tk, embedding: OllamaEmbeddings, model_name: str, chunk_size: int):
        """ Init, it inits...."""
        self.root = root
        self.embedding = embedding
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.root.title("Prototype")
        self.chat_history = []
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        
        # self.mapping = {"A": "llama3:latest", "B": "llama3.2:1b", "C": "llama3.2:latest"} if random.uniform(0,1) > 1.5 else {"A": "llama3.2:latest", "B": "llama3.2:latest", "C": "llama3.2:latest"}
        self.model = ChatOllama(model=self.model_name)
        
        #Chat History Display
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=20, width=80)
        self.chat_display.pack(pady=10)

        #Input Box
        self.user_input = tk.Entry(root, width=70)
        self.user_input.pack(side=tk.LEFT, padx=5)
        self.user_input.bind("<Return>", self.on_enter_pressed)
        self.send_button = tk.Button(root, text="Send", command=self.ask_question)
        self.send_button.pack(side=tk.LEFT, padx=5)
        

        #Close button
        self.close_button = tk.Button(root, text="Close and Evaluate", command=self.close_and_upload)
        self.close_button.pack(pady=10)

    def on_enter_pressed(self, event):
        self.ask_question()

    def close_and_upload(self):
        """ Function to pose questionnaire after testing the different models, saved to xlsx in ../Results/Questionnaire"""
        #Create a new window for the questionnaire
        questionnaire_window = tk.Toplevel(self.root)
        questionnaire_window.title("Questionnaire")
        
        questions = [
            "Who do you think commited the murder?",
            "Why did this person commit this murder?",
            "Please evaluate your assistant qualitatively in terms of how helpful it was."
        ]
        
        answers = []
        
        #Function to save answers and close the window
        def submit_answers():
            """ Answer submitted to a xlsx file in a directory. Filename always unique and indicates of same of different models were used."""
            for entry in answer_entries:
                answers.append(entry.get())
            
            #Save to Excel
            data = {'Question': questions, 'Answer': answers}
            df = pd.DataFrame(data)
            save_path = "../Results/Questionnaire/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            _, _, files = next(os.walk(save_path))
            idx = len(files) + 1
            save_file = save_path + str(idx) + "SameModelquestionnaire_answers.xlsx" if len(set(self.mapping.values())) == 1 else save_path + str(idx) + "questionnaire_answers.xlsx"
            df.to_excel(save_file, index=False)
            
            messagebox.showinfo("Thanks!", "Your answers have been saved!")
            questionnaire_window.destroy()
            self.root.destroy()
        
        #Create a label and entry field for each question
        answer_entries = []
        for _, question in enumerate(questions):
            question_label = tk.Label(questionnaire_window, text=question, anchor='w', justify='left', wraplength=400)
            question_label.pack(pady=(10, 2), padx=10, anchor='w')
            answer_entry = tk.Entry(questionnaire_window, width=50)
            answer_entry.pack(padx=10, pady=5)
            answer_entries.append(answer_entry)
        
        #Add a submit button
        submit_button = tk.Button(questionnaire_window, text="Submit", command=submit_answers)
        submit_button.pack(pady=20)


    def summarize_history(self):
        """Generate a summary of the chat history."""
        if not self.chat_history:
            messagebox.showinfo("Info", "Chat history is empty. Start a conversation first.")
            return
        
        try:
            #Concatenate chat history into a single string for 
            history_text = "\n".join([f"You: {q}\nAssistant: {a}" for q, a in self.chat_history])
            
            #Use the model to summarize the history
            summary = self.model.invoke(f"Please summarize the following conversation:\n{history_text}")
            
            # self.log_chat("System", f"Chat history summarized.")
            return summary.content

        except Exception as e:
            self.log_chat("System", f"Error summarizing chat history: {e}")
            messagebox.showerror("Error", f"Error summarizing chat history: {e}")
            return None


    def log_chat(self, role, message):
        """Log messages in the chat display."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{role}: {message}\n")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def ask_question(self):
        # Custom prompt for Murder Mystery scenario
        custom_prompt = f"""
                You are a helpful assistant that will play a game with a human that has access to information
                concerning a murder mysterie. Together you will solve this mysterie based on the documents this 
                person has, you need to ask questions about the documents and have a discussion with the user to get to a conclusion on who comitted the murder.
                Always be polite, here is the chat history, in this history you are "Botanic" and human is "user" 
                
                Ask a question that could help solve this mystery:
            """
        
        # Generate response from the LLM
        result = self.model.invoke(custom_prompt)
        bot_response = result.content

        # Log and update chat history
        self.log_chat("Botanic", bot_response)

        user_message = self.user_input.get()

        self.log_chat("User", user_message)
        self.user_input.delete(0, tk.END)

        self.chat_history.append((user_message, bot_response))
        self.summarize_history()

#Create and run the app
if __name__ == "__main__":
    """Main loop"""
    model_name = "llama3.2:1b"
    embedding = "nomic-embed-text"
    chunk_size = 512
    root = tk.Tk()
    app = ChatApp(root, embedding, model_name, chunk_size)
    root.mainloop()

#llama3:latest              365c0bd3c000    4.7 GB    params: 8,03B
#llama3.2:1b                baf6a787fdff    1.3 GB    params: 1B
#llama3.2:latest            a80c4f17acd5    2.0 GB    params: 3,21B
#Embedding
#nomic-embed-text:latest    0a109f422b47    274 MB         