import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from time import sleep
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
    # TODO: Enter for send
    
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
        
        #self.chat_summary = '' To be continued
        
        #Model selection and initialization
        self.model_label = tk.Label(root, text="Select Model:")
        self.model_label.pack()
        self.model_options = ["A", "B", "C"] 
        self.mapping = {"A": "llama3:latest", "B": "llama3.2:1b", "C": "llama3.2:latest"} if random.uniform(0,1) > 0.5 else {"A": "llama3:latest", "B": "llama3:latest", "C": "llama3:latest"}
        self.selected_model = tk.StringVar(value="A")
        self.model_name = self.mapping[self.selected_model.get()]
        self.model_dropdown = tk.OptionMenu(root, self.selected_model, *self.model_options, command=self.update_model)
        self.model_dropdown.pack(pady=5)
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
        
        #File Upload Button
        self.upload_button = tk.Button(root, text="Upload File", command=self.upload_file)
        self.upload_button.pack(pady=10)

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
            "Did you notice a difference in quality of results given by the different models, if yes what was it?",
            "Which model did you find the easiest to use for completing the task, and why?",
            "If you have to choose one of the models for future use, which one would you choose."
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


    def update_model(self, selected_option):
        """Update the model based on dropdown selection."""
        #Map the user-facing option to the internal model name
        self.model_name = self.mapping[selected_option]
        self.model = ChatOllama(model=self.model_name)
        self.log_chat("System", f"Model switched to: {list(self.mapping.keys())[list(self.mapping.values()).index(self.model_name)]}")
        self.chat_history = []
        self.log_chat("System", "Chat history reset!")

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
            
            self.log_chat("System", f"Chat history summarized.")
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

    def upload_file(self):
        """Handle file upload and initialize vectorstore."""
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")])
        if file_path:
            self.log_chat("System", f"Processing file: {file_path}")
            try:
                docs = document_parsing(file_path, chunk_size=self.chunk_size)
                local_embeddings = OllamaEmbeddings(model=self.embedding)
                text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size/10)
                texts = text_splitter.split_documents(docs)
                self.vectorstore = Chroma.from_documents(documents=texts, embedding=local_embeddings)
                retriever = self.vectorstore.as_retriever(search_kwargs={'k': 8})
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    self.model,
                    retriever,
                    get_chat_history=lambda h : h,
                    return_source_documents=True
                )
                self.log_chat("System", "File uploaded and processed successfully.")
            except Exception as e:
                self.log_chat("System", f"Error processing file: {e}")
                messagebox.showerror("Error", f"Could not process the file: {e}")

    def ask_question(self):
        """Handle user questions and provide responses."""
        if not self.qa_chain:
            messagebox.showwarning("Warning", "Please upload a file before asking questions.")
            return

        user_message = self.user_input.get()
        if not user_message.strip():
            return

        self.log_chat("You", user_message)
        self.user_input.delete(0, tk.END)

        try:
            result = self.qa_chain({'question': user_message, 'chat_history': self.chat_history})
            
            bot_response = result['answer']
            sleep(random.randint(0,5))
            self.log_chat("Botanic", bot_response)

            self.chat_history.append((user_message, bot_response))

        except Exception as e:
            self.log_chat("System", f"Error answering question: {e}")
            messagebox.showerror("Error", f"Error answering question: {e}")

#Create and run the app
if __name__ == "__main__":
    """Main loop"""
    model_name = "Select From List"
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